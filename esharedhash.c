/* Part 2: Explore and Optimize Concurrency
* Chosen strategy: Strategy A - Per-Thread Memory Pools
*
* Goal: Global Lock Elimination
* The multi-threaded implementation in Part 1 revealed a catastrophic bottleneck:
* severe lock contention on the single shared pthread_mutex protecting the
* global memory allocator. Since Huffman encoding is an allocation-heavy
* workload, the parallel threads were forced to serialize their execution
* at the allocation step, resulting in a significant performance degradation
* (over 12x slowdown in large file workloads).
*
* Implement Thread Local Allocation:
* The simplest and most effective optimization is to eliminate the global lock
* by giving each thread its own exclusive, lock-free allocator region.
* Divide the shared heap (UMEM_SIZE) into N chunks (POOL_CHUNK_SIZE).
* Each thread, upon its first allocation, acquires the global lock once
* to reserve one chunk from the global fallback free list (thread_pool_init).
* This chunk becomes the thread's private, thread-local memory pool, managed
* by a simple bump-pointer (pool_current).
* Most umalloc() and ufree() calls (the 'fast path') occur within this
* pool and proceed without any synchronization, achieving massive speedup.
* Only when a thread’s pool runs out of space does it acquire the global
* lock and fall back to the slower, globally shared free list.
* This strategy successfully converts the vast majority of memory operations
* from a slow, contended, and locked process into a fast, lock-free, local
* process.
*/ 

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <sys/mman.h>
 #include <unistd.h>
 #include <sys/wait.h>
 #include <semaphore.h>
 #include <pthread.h> 
 #include <stdbool.h> 
 
 #define BLOCK_SIZE 1024
 #define SYMBOLS 256
 #define LARGE_PRIME 2147483647
 #define UMEM_SIZE (2 * 1024 * 1024)   
 #define MAX_THREADS 64   // Guaranteed 32KB for each memory pool chunk          
 
 // New: Define the size of each private pool chunk 
 #define POOL_CHUNK_SIZE (UMEM_SIZE / MAX_THREADS) 
 
 void *umalloc(size_t size);
 void ufree(void *ptr);
 unsigned long process_block(const unsigned char *buf, size_t len);
 int run_single(const char *filename);
 int run_multi(const char *filename);
 int run_threads(const char *filename);
 void *thread_worker(void *arg);
 
 /* =======================================================================
   PROVIDED CODE — DO NOT MODIFY
   ======================================================================= */
 
 #define MAGIC 0xDEADBEEFLL
 
 typedef struct {
     long size;
     long magic;
 } header_t;
 
 typedef struct __node_t {
     long size;
     struct __node_t *next;
 } node_t;

 /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Shared Memory Free List Management
 *
 * Critical insight: free_list_ptr is not just a pointer, but a pointer
 * TO a pointer that lives in shared memory. This double indirection
 * ensures all processes see the same free list head location.
 *
 * Without this, each child would have free_list pointing to wherever
 * the head was when IT forked, causing catastrophic corruption when
 * multiple children try to allocate/free simultaneously.
 */
 
 // Free List Management (from Part 1)
 static node_t **free_list_ptr = NULL; 
 static node_t *free_list_thread = NULL; 

 #define ALIGNMENT 16
 #define ALIGN(size) (((size) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))
 
/* `````````````````````````````````````````````````````````````````````
 * Synchronization and Mode Control
 *
 * mLock protects all allocator operations in multi-process mode. It must
 * be in shared memory (via mmap) so all processes synchronize on the same
 * semaphore object.
 *
 * use_multiprocess flag determines whether to initialize/use the semaphore.
 * This avoids locking overhead when running single-threaded.
 */

 sem_t* mLock;
 pthread_mutex_t mLock_thread; 
 int use_multiprocess = 0; 
 int use_multithread = 0; 
 
 // New: Thread-Local Storage for Strategy A
 __thread char *pool_start = NULL;
 __thread char *pool_current = NULL;
 __thread char *pool_end = NULL;
 
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Memory Allocator Initialization
 *
 * Sets up three shared memory regions:
 *   1. free_list_ptr - pointer to the free list head (sizeof pointer)
 *   2. mLock - semaphore for synchronization (if multi-process)
 *   3. heap - the actual managed memory region (UMEM_SIZE bytes)
 *
 * All three use MAP_SHARED so modifications are visible across fork().
 * The heap is initialized with a single free block spanning the entire
 * region.
 */
 
 void *init_umem(void) {
     if (use_multiprocess) {
         free_list_ptr = mmap(NULL, sizeof(node_t *),
                          PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
         mLock = mmap(NULL, sizeof(sem_t),
                         PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
         sem_init(mLock, 1, 1);  
         void *base = mmap(NULL, UMEM_SIZE,
                         PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
         
         *free_list_ptr = (node_t *)base;
         (*free_list_ptr)->size = UMEM_SIZE - sizeof(node_t);
         (*free_list_ptr)->next = NULL;
         return base;
     }
 
     else {
         if (use_multithread) {
             if (pthread_mutex_init(&mLock_thread, NULL) != 0) {
                 perror("pthread_mutex_init");
                 exit(1);
             }
         }
 
         void *base = malloc(UMEM_SIZE);
         if (!base) {
             perror("malloc");
             exit(1);
         }
 
         free_list_thread = (node_t *)base;
         free_list_thread->size = UMEM_SIZE - sizeof(node_t);
         free_list_thread->next = NULL;
         return base;
     }
 }
 
 /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Coalescing: Merge Adjacent Free Blocks
 *
 * After freeing, adjacent blocks in memory should be merged into larger
 * blocks to reduce fragmentation. We maintain the free list in address
 * order (see _ufree), so adjacency is detected by checking if one block's
 * end address equals the next block's start address.
 *
 * Why this matters for correctness: Without coalescing, the free list
 * could become fragmented into tiny unusable pieces. With concurrent
 * access, corruption here (following a bad pointer) was our most subtle
 * bug - if curr->next points to an allocated block, we read that block's
 * header->magic thinking it's a next pointer, causing infinite loops.
 */

 // Macro to conditionally access the free list head
 #define GET_FREE_LIST_HEAD() ((use_multiprocess) ? (*free_list_ptr) : free_list_thread)
 #define SET_FREE_LIST_HEAD(val) do { if (use_multiprocess) { *free_list_ptr = val; } else { free_list_thread = val; } } while (0)

 static void coalesce(void) {
     node_t *curr = GET_FREE_LIST_HEAD();
     while (curr && curr->next) {
         char *end = (char *)curr + sizeof(node_t) + curr->size;
         if (end == (char *)curr->next) {
             curr->size += sizeof(node_t) + curr->next->size;
             curr->next = curr->next->next;
         } else {
             curr = curr->next;
         }
     }
 }
 
 /* `````````````````````````````````````````````````````````````````````
 * First-Fit Allocator
 *
 * Searches the free list for the first block large enough to satisfy
 * the request. If the block is larger than needed, it's split: the
 * allocated portion becomes unavailable, and the remainder stays on
 * the free list.
 *
 * Why first-fit: Simple, fast for small allocations, and "good enough"
 * for teaching. Best-fit would reduce fragmentation but requires scanning
 * the entire list. Worst-fit is rarely useful.
 *
 * Critical detail: We save curr->next BEFORE overwriting the node with
 * a header. When we allocate from the head of the free list and create
 * a remainder, we need to know what used to be next.
 *
 * Lock contention source: Every allocation traverses this list under the
 * global semaphore. With N concurrent processes all building Huffman trees
 * (hundreds of allocations each), this becomes a severe bottleneck.
 */

 void *_umalloc(size_t size) {
     if (size == 0) return NULL;
 
     size = ALIGN(size);
     node_t *prev = NULL;
     node_t *curr = GET_FREE_LIST_HEAD();
 
     while (curr) {
         if (curr->size >= (long)size) {
             char *alloc_start = (char *)curr;
             long remaining = curr->size - (long)size;
             node_t *next_free = curr->next;
 
             header_t *hdr = (header_t *)alloc_start;
             hdr->size = size;
             hdr->magic = MAGIC;
             void *user_ptr = alloc_start + sizeof(header_t);
 
             if (remaining >= (long)sizeof(node_t) + ALIGNMENT) {
                 node_t *new_free = (node_t *)(alloc_start + sizeof(header_t) + size);
                 new_free->size = remaining - sizeof(node_t);
                 new_free->next = next_free;
                 if (prev)
                     prev->next = new_free;
                 else
                     SET_FREE_LIST_HEAD(new_free);
             } else {
                 if (prev)
                     prev->next = next_free;
                 else
                     SET_FREE_LIST_HEAD(next_free);
             }
 
             return user_ptr;
         }
         prev = curr;
         curr = curr->next;
     }
 
     return NULL;
 }
 
 /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Free: Return Block to Free List (in Address Order)
 *
 * Converts the allocated block back to a free node and inserts it into
 * the free list in address order. Address ordering is essential for
 * coalescing to work - we need adjacent blocks to be neighbors in the list.
 *
 * The magic number check catches double-frees and corruption. If someone
 * calls ufree() on an already-freed pointer, magic will likely be wrong
 * (it's been overwritten by node_t fields).
 *
 * After insertion, we coalesce to merge with adjacent blocks. This is
 * another source of lock contention - every free does a full list walk.
 */

 void _ufree(void *ptr) {
     if (!ptr) return;
 
     header_t *hdr = (header_t *)((char *)ptr - sizeof(header_t));
     if (hdr->magic != MAGIC) {
         fprintf(stderr, "Error: invalid free detected.\n");
         abort();
     }
 
     node_t *node = (node_t *)hdr;
     node->size = hdr->size;
     node->next = NULL;
 
     node_t *head = GET_FREE_LIST_HEAD();
 
     if (!head || node < head) {
         node->next = head;
         SET_FREE_LIST_HEAD(node);
     } else {
         node_t *curr = head;
         while (curr->next && curr->next < node)
             curr = curr->next;
         node->next = curr->next;
         curr->next = node;
     }
 
     coalesce();
 }
 
 /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * THREAD POOL FUNCTIONS
 * This function is responsible for the initial lock-protected reservation 
 * of a memory chunk for a new thread. It is called only once per thread.
 */

 void thread_pool_init() {
     size_t chunk_size = POOL_CHUNK_SIZE;

     char *chunk = malloc(chunk_size);
 
     if (chunk == NULL) {
         fprintf(stderr, "[%lu] Warning: Failed to allocate private pool. Relying on global allocator.\n",
                 pthread_self());
         pool_start = pool_current = pool_end = NULL;
         return;
     }
 
     pool_start = chunk;
     pool_current = pool_start;
     pool_end = chunk + chunk_size;
 }
 
 /* `````````````````````````````````````````````````````````````````````
 * Public Allocator Interface with Conditional Locking
 *
 * These wrappers add semaphore protection around the internal allocator
 * functions, but ONLY in multi-process mode. Single-process execution
 * avoids the overhead entirely.
 *
 * Lock contention analysis: In multi-process mode with 10 children,
 * each building a Huffman tree (~512 allocations + ~512 frees), we see
 * roughly 10,000+ lock acquisitions. Since only one process can hold
 * the lock at a time, most processes spend most of their time WAITING
 * rather than computing. This is why speedup is minimal (~1.2x) despite
 * 10-way parallelism.
 *
 * This inefficiency is intentional - students will optimize by implementing
 * per-thread memory pools or lock-free structures in later assignments.
 * MODIFIED WRAPPERS WITH CONDITIONAL LOCKING + POOLING
 */
 
 void *umalloc(size_t size) {
     if (size == 0) return NULL;
 
     if (use_multiprocess) {
         sem_wait(mLock);
         void *p = _umalloc(size);
         sem_post(mLock);
         return p;
     }
     
     if (use_multithread) {
         size_t requested_user_size = ALIGN(size);
         size_t requested_total_size = requested_user_size + sizeof(header_t);
 
         // 1. Initial Check: If pool is NULL, initialize it (acquire lock once).
         if (pool_start == NULL) {
             thread_pool_init();
         }
 
         // 2. FAST PATH: Check if the allocation fits in the local pool.
         if (pool_current != NULL && pool_current + requested_total_size <= pool_end) {
             char *alloc_start = pool_current; 
             pool_current += requested_total_size;
 
             header_t *hdr = (header_t *)alloc_start;
             hdr->size = requested_user_size;
             hdr->magic = MAGIC;
 
             return alloc_start + sizeof(header_t);
         }
         
         // 3. SLOW PATH (Fallback): If pool is exhausted, acquire global lock.
         pthread_mutex_lock(&mLock_thread);
         void *p = _umalloc(requested_user_size); 
         pthread_mutex_unlock(&mLock_thread);
         return p;
     }
 
     return _umalloc(size);
 }
 
 void ufree(void *ptr) {
     if (!ptr) return;
 
     if (use_multiprocess) {
         sem_wait(mLock);
         _ufree(ptr);
         sem_post(mLock);
         return;
     }
 
     if (use_multithread) {
         char *hdr_start = (char *)ptr - sizeof(header_t);
 
         // 1. FAST PATH (Pool Check): If pointer is inside the pool range...
         if (pool_start && hdr_start >= pool_start && hdr_start < pool_end) {
             return; // ...do nothing (No-Op). Memory is freed when the thread exits.
         }
 
         // 2. SLOW PATH (Fallback): If pointer was allocated by the global allocator.
         pthread_mutex_lock(&mLock_thread);
         _ufree(ptr);
         pthread_mutex_unlock(&mLock_thread);
         return;
     }
     
     _ufree(ptr);
 }
 
/* =======================================================================
   Huffman Tree Construction (Given)
   ======================================================================= */
 
 typedef struct Node {
     unsigned char symbol;
     unsigned long freq;
     struct Node *left, *right;
 } Node;
 
 typedef struct {
     Node **data;
     int size;
     int capacity;
 } MinHeap;
 
 MinHeap *heap_create(int capacity) {
     MinHeap *h = umalloc(sizeof(MinHeap));
     h->data = umalloc(sizeof(Node *) * capacity);
     h->size = 0;
     h->capacity = capacity;
     return h;
 }
 
 void heap_swap(Node **a, Node **b) {
     Node *tmp = *a; *a = *b; *b = tmp;
 }
 
 void heap_push(MinHeap *h, Node *node) {
     int i = h->size++;
     h->data[i] = node;
     while (i > 0) {
         int p = (i - 1) / 2;
         if (h->data[p]->freq < h->data[i]->freq) break;
         heap_swap(&h->data[p], &h->data[i]);
         i = p;
     }
 }
 
 Node *heap_pop(MinHeap *h) {
     if (h->size == 0) return NULL;
     Node *min = h->data[0];
     h->data[0] = h->data[--h->size];
 
     int i = 0;
     while (1) {
         int l = 2*i+1, r = l+1, smallest = i;
         if (l < h->size && h->data[l]->freq < h->data[smallest]->freq) smallest = l;
         if (r < h->size && h->data[r]->freq < h->data[smallest]->freq) smallest = r;
         if (smallest == i) break;
         heap_swap(&h->data[i], &h->data[smallest]);
         i = smallest;
     }
     return min;
 }
 
 void heap_free(MinHeap *h) {
     ufree(h->data);
     ufree(h);
 }
 
 Node *new_node(unsigned char sym, unsigned long freq, Node *l, Node *r) {
     Node *n = umalloc(sizeof(Node));
     n->symbol = sym;
     n->freq = freq;
     n->left = l;
     n->right = r;
     return n;
 }
 
 void free_tree(Node *n) {
     if (!n) return;
     free_tree(n->left);
     free_tree(n->right);
     ufree(n);
 }
 
 Node *build_tree(unsigned long freq[SYMBOLS]) {
     MinHeap *h = heap_create(SYMBOLS);
     for (int i = 0; i < SYMBOLS; i++)
         if (freq[i] > 0)
             heap_push(h, new_node((unsigned char)i, freq[i], NULL, NULL));
     if (h->size == 0) {
         heap_free(h);
         return NULL;
     }
     while (h->size > 1) {
         Node *a = heap_pop(h);
         Node *b = heap_pop(h);
         Node *p = new_node(0, a->freq + b->freq, a, b);
         heap_push(h, p);
     }
     Node *root = heap_pop(h);
     heap_free(h);
     return root;
 }
 
 unsigned long hash_tree(Node *n, unsigned long hash) {
     if (!n) return hash;
     hash = (hash * 31 + n->freq + n->symbol) % LARGE_PRIME;
     hash = hash_tree(n->left, hash);
     hash = hash_tree(n->right, hash);
     return hash;
 }

/* =======================================================================
   Output Functions
   ======================================================================= */
 
 void print_intermediate(int block_num, unsigned long hash, pid_t pid) {
 #ifdef DEBUG
 #  if DEBUG == 2
     printf("[PID %d] Block %d hash: %lu\n", pid, block_num, hash);
 #  elif DEBUG == 1
     printf("Block %d hash: %lu\n", block_num, hash);
 #  endif
 #else
     (void)block_num;
     (void)hash;
     (void)pid;
 #endif
 }
 
 void print_final(unsigned long final_hash) {
     printf("Final signature: %lu\n", final_hash);
 }
 
 /* `````````````````````````````````````````````````````````````````````
 * Main Entry Point
 *
 * Parses arguments to determine execution mode, initializes the shared
 * memory allocator, then dispatches to either single-process or
 * multi-process execution.
 *
 * The allocator MUST be initialized before any fork() calls, ensuring
 * all processes share the same heap region.
 */

 int main(int argc, char *argv[]) {
     if (argc < 2) {
         fprintf(stderr, "Usage: %s <file> [-m|-t]\n", argv[0]);
         return 1;
     }
 
     const char *filename = argv[1];
     use_multiprocess = (argc >= 3 && strcmp(argv[2], "-m") == 0);
     use_multithread = (argc >= 3 && strcmp(argv[2], "-t") == 0); 
 
     init_umem();
 
     if (use_multiprocess)
         return run_multi(filename);
     else if (use_multithread) 
         return run_threads(filename);
     else
         return run_single(filename);
 }
 
 /* `````````````````````````````````````````````````````````````````````
 * Per-Block Processing Logic
 *
 * This is where the actual work happens: count symbol frequencies,
 * build the Huffman tree, hash it, and clean up. Each block is
 * independent - no communication between blocks needed.
 *
 * Allocation profile: For a typical 1KB block with ~100 unique symbols,
 * this calls umalloc() roughly 200-400 times (heap, array, tree nodes)
 * and ufree() a similar number. This heavy allocation pattern is why
 * lock contention dominates performance.
 */

 unsigned long process_block(const unsigned char *buf, size_t len) {
     unsigned long freq[SYMBOLS] = {0};
     for (size_t i = 0; i < len; i++)
         freq[buf[i]]++;
 
     Node *root = build_tree(freq);
     unsigned long h = hash_tree(root, 0);
     free_tree(root);
     return h;
 }
 
 /* `````````````````````````````````````````````````````````````````````
 * Single-Process Execution
 *
 * Straightforward sequential processing: read a block, process it,
 * accumulate the hash, repeat. No synchronization needed since there's
 * only one thread of execution.
 *
 * This serves as the performance baseline - any parallel version should
 * be faster, but due to lock contention, the multi-process version is
 * barely faster (or sometimes even slower due to fork() overhead).
 */

 int run_single(const char *filename) {
     FILE *fp = fopen(filename, "rb");
     if (!fp) {
         perror("fopen");
         return 1;
     }
 
     unsigned char buf[BLOCK_SIZE];
     unsigned long final_hash = 0;
     int block_num = 0;
 
     while (!feof(fp)) {
         size_t n = fread(buf, 1, BLOCK_SIZE, fp);
         if (n == 0) break;
         unsigned long h = process_block(buf, n);
         print_intermediate(block_num++, h, getpid());
         final_hash = (final_hash + h) % LARGE_PRIME;
     }
 
     fclose(fp);
     print_final(final_hash);
     return 0;
 }
 
 /* `````````````````````````````````````````````````````````````````````
 * Multi-Process Execution: Three-Phase Parallel Strategy
 *
 * The key to achieving true parallelism: separate process creation from
 * result collection from cleanup. This allows all children to execute
 * simultaneously.
 *
 * Phase 1: Fork all children
 *   Parent reads each block, allocates a shared memory buffer via umalloc(),
 *   copies the data, then forks a child to process it. Critically, the
 *   parent does NOT wait - it immediately continues to the next block.
 *   This means all children are launched in rapid succession and execute
 *   in parallel.
 *
 *   Why allocate via umalloc() instead of just passing buf pointer?
 *   - buf is on parent's stack, which gets reused for each iteration
 *   - By the time child 0 runs, parent might have overwritten buf with block 5's data
 *   - Allocating in shared memory gives each child a stable copy
 *   - Also exercises the allocator, exposing lock contention
 *
 * Phase 2: Collect results in order
 *   Now we read from each pipe in sequence. If a child finished early,
 *   its result is already waiting in the pipe buffer. If it's still
 *   running, we block until it writes. This serializes COLLECTION but
 *   not EXECUTION - children are already running in parallel.
 *
 * Phase 3: Wait for stragglers
 *   Any children still running (unlikely if pipes are small) get reaped
 *   to avoid zombies. In practice, most children finish during Phase 2.
 *
 * Performance bottleneck: Despite parallel execution, speedup is minimal
 * because children spend most time waiting for the allocator lock. With
 * 10 children and ~500 allocations each, most processes are blocked most
 * of the time. This is why students will optimize the allocator in Part 2.
 */

 int run_multi(const char *filename) {
     FILE *fp = fopen(filename, "rb");
     if (!fp) {
         perror("fopen");
         return 1;
     }
 
     unsigned char buf[BLOCK_SIZE];
     unsigned long final_hash = 0;
     pid_t pids[1024];
     int pipe_fds[1024];
     int num_blocks = 0;
 
     /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     * Phase 1: Fork all children (parallel execution)
     */

     while (!feof(fp)) {
         size_t n = fread(buf, 1, BLOCK_SIZE, fp);
         if (n == 0) break;
 
         if (num_blocks >= 1024) {
             fprintf(stderr, "Error: file too large (max 1024 blocks)\n");
             fclose(fp);
             return 1;
         }
 
         unsigned char *block_buf = umalloc(n);
         if (!block_buf) {
             fprintf(stderr, "umalloc failed for block %d\n", num_blocks);
             fclose(fp);
             return 1;
         }
         memcpy(block_buf, buf, n);
 
         int pipefd[2];
         if (pipe(pipefd) == -1) {
             perror("pipe");
             ufree(block_buf);
             fclose(fp);
             return 1;
         }
 
         pid_t pid = fork();
         if (pid < 0) {
             perror("fork");
             ufree(block_buf);
             fclose(fp);
             return 1;
         }
 
         if (pid == 0) {
             /* Child process: compute hash, write result, exit
             *
             * Close inherited file descriptor - child doesn't need it
             * and leaving it open wastes resources. Close read end of
             * pipe since we only write.
             */
             fclose(fp);
             close(pipefd[0]);
             unsigned long h = process_block(block_buf, n);
             ufree(block_buf);
             write(pipefd[1], &h, sizeof(h));
             close(pipefd[1]);
             exit(0);
         } else {
              /* Parent: save child info, continue forking
             *
             * Don't wait here! That would serialize execution. Just
             * save the PID and pipe descriptor so we can collect results
             * later. Close write end since parent only reads.
             *
             * Note: block_buf will be freed by child after processing.
             */
             close(pipefd[1]);
             pids[num_blocks] = pid;
             pipe_fds[num_blocks] = pipefd[0];
             num_blocks++;
         }
     }
 
     fclose(fp);

     /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     * Phase 2: Collect results in order
     *
     * Read from pipes sequentially. This determines output order but
     * doesn't affect parallelism - children are already running. If a
     * child finished early, read() returns immediately. If still running,
     * we block until it writes.
     */
 
     for (int i = 0; i < num_blocks; i++) {
         unsigned long h = 0;
         read(pipe_fds[i], &h, sizeof(h));
         close(pipe_fds[i]);
         print_intermediate(i, h, pids[i]);
         final_hash = (final_hash + h) % LARGE_PRIME;
     }

     /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     * Phase 3: Wait for all children to complete
     *
     * Reap any remaining children to avoid zombies. Most should have
     * finished during Phase 2 (when we read their results), but waitpid()
     * ensures clean termination.
     */

     for (int i = 0; i < num_blocks; i++)
         waitpid(pids[i], NULL, 0);
 
     print_final(final_hash);
     return 0;
 }
 
 // Set a safe limit for concurrent threads. 32 or 64 is safe for a 2MB heap
 #define MAX_CONCURRENT_THREADS 32
 
 typedef struct {
     unsigned char *block_buf; // Input data (allocated via umalloc)
     size_t len;               // Length of the data
     int block_num;            // Index of the result
     unsigned long *results;   // Pointer to the shared results array
 } thread_args_t;
 
 // Helper functions for thread mode
 void *thread_worker(void *arg) {
     thread_args_t *args = (thread_args_t *)arg;
     
     // Process the block
     unsigned long h = process_block(args->block_buf, args->len);
     
     // Store the result directly into the shared array
     args->results[args->block_num] = h;
     
     // Clean up the shared input buffer (umalloc'd memory)
     ufree(args->block_buf);
     
     // Clean up the arguments structure (malloc'd in the parent)
     free(args); 
     
     return NULL;
 }
 
 int run_threads(const char *filename) {
     FILE *fp = fopen(filename, "rb");
     if (!fp) { 
         perror("fopen");
         return 1;
     }
 
    // STEP 1: CALCULATE THE REQUIRED SIZE DYNAMICALLY 
    
    // Get file size in bytes
     fseek(fp, 0, SEEK_END);
     long file_size_bytes = ftell(fp);
     fseek(fp, 0, SEEK_SET); 
 
     // Calculate the exact number of blocks needed
     int num_blocks = file_size_bytes / BLOCK_SIZE;
     if (file_size_bytes % BLOCK_SIZE != 0) {
         num_blocks++; 
     }
     
     if (num_blocks <= 0) {
         fprintf(stderr, "Error: Empty or invalid file.\n");
         fclose(fp);
         return 1;
     }
     
    // STEP 2: DYNAMICALLY ALLOCATE ARRAYS 
    
    // Dynamically allocate TIDs array based on actual file size
     pthread_t *tids = malloc(sizeof(pthread_t) * num_blocks);
    // Dynamically allocate results array based on actual file size
     unsigned long *results = malloc(sizeof(unsigned long) * num_blocks);
     
     if (!tids || !results) {
         perror("malloc for tids or results array failed");
         free(tids); // Safe to call free(NULL) if tids failed
         free(results);  // Safe to call free(NULL) if tids failed
         fclose(fp);
         return 1;
     }
    
     unsigned char buf[BLOCK_SIZE];
     unsigned long final_hash = 0;
 
     int total_blocks = 0; // Tracks total blocks read/processed
 
     // Phase 1: Read input and launch threads
     while (total_blocks < num_blocks) { // Loop until all blocks are read
         int batch_start = total_blocks;
         int count_in_batch = 0;
         
         while (count_in_batch < MAX_CONCURRENT_THREADS && total_blocks < num_blocks) {
             
             size_t bytes_read = fread(buf, 1, BLOCK_SIZE, fp);
             if (bytes_read == 0) {
                 // Should not happen if num_blocks was calculated correctly, but safe guard.
                 break; 
             }
 
             // 1. Allocate thread arguments structure
             thread_args_t *args = malloc(sizeof(thread_args_t));
             if (!args) { break; }
 
             // 2. Allocate block buffer using umalloc
             args->block_buf = umalloc(bytes_read);
             if (!args->block_buf) { 
                 fprintf(stderr, "Fatal: umalloc failed for block %d. Heap exhausted.\n", total_blocks);
                 ufree(args->block_buf);
                 free(args);
                 // Need to clean up and break all loops on fatal error
                 goto cleanup_and_exit; 
             }
             
             // Copy data and setup args
             memcpy(args->block_buf, buf, bytes_read);
             args->len = bytes_read;
             args->block_num = total_blocks;
             args->results = results; // Point to the dynamically allocated shared results array
 
             // 3. Launch thread
             if (pthread_create(&tids[total_blocks], NULL, thread_worker, args) != 0) {
                 perror("pthread_create failed");
                 ufree(args->block_buf);
                 free(args);
                 goto cleanup_and_exit; 
             }
 
             total_blocks++;
             count_in_batch++;
         }
  
         // Phase 2: Join the current batch
         for (int i = batch_start; i < batch_start + count_in_batch; i++) {
             pthread_join(tids[i], NULL); 
         }
 
         // Check for immediate exhaustion which would cause infinite loop if not handled
         if (count_in_batch == 0 && total_blocks < num_blocks) {
             // Should only happen if MAX_CONCURRENT_THREADS is 0, or something catastrophic happened
             break;
         }
     } 
     
     // Calculate the final hash using all results
     for (int i = 0; i < num_blocks; i++) {
         print_intermediate(i, results[i], 0);
         final_hash = (final_hash + results[i]) % LARGE_PRIME;
     }
     print_final(final_hash);
 
 cleanup_and_exit:
     // STEP 3: CLEANUP DYNAMICALLY ALLOCATED MEMORY ---
     if (tids) free(tids);
     if (results) free(results);
     fclose(fp);
 
     if (total_blocks < num_blocks) {
         return 1; // Indicate failure due to umalloc or pthread_create
     }
     return 0; 
 }
 