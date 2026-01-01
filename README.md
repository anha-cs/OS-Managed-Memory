# OS-Managed-Memory
Brief introduction: A concurrent memory allocator that explores the transition from multi-process shared memory to multi-threaded execution. This project implements a thread-safe, first-fit heap allocator using pthreads and optimizes performance by implementing Per-Thread Memory Pools to eliminate global lock contention.


---

# Canvas OS Project: Managed Memory with Threads & Concurrency

## Project Overview

This project explores the evolution of memory management in concurrent systems. It transitions from a **multi-process model**—where a shared heap is managed via `mmap()` and semaphores—to a **multi-threaded model** using `pthreads`.

The core application is a Huffman-based hashing program that processes input files in 1KB blocks. The project concludes with an experimental allocator that utilizes **Per-Thread Memory Pools** to maximize parallel throughput.

---

## 🛠 Compilation & Usage

### 1. Basic Compilation

To compile the standard threaded version (Part 1):

```bash
gcc -Wall -Wextra -pthread -o sharedhash sharedhash.c

```

To compile the experimental optimized version (Part 2):

```bash
gcc -Wall -Wextra -pthread -o esharedhash esharedhash.c

```

### 2. Execution Modes

* **Single-threaded:** `./sharedhash <filename>`
* **Multi-threaded (-t):** `./sharedhash <filename> -t` (Uses `pthread_create`)
* **Multi-process (-m):** `./sharedhash <filename> -m` (Legacy process mode)

---

## 🏗 System Architecture

### Process vs. Thread Memory Model

In the threaded version, threads inhabit a single address space. This allows for direct data sharing without the overhead of `mmap()` or Inter-Process Communication (IPC) like pipes.

### The Allocator Evolution

1. **Baseline (sharedhash.c):** Uses a coarse-grained global `pthread_mutex_t`. Every call to `umalloc()` and `ufree()` locks the entire heap, serializing execution.
2. **Optimized (esharedhash.c):** Implements **Per-Thread Memory Pools**.
* The heap is divided into  regions (where  is the number of threads).
* Each thread is assigned a private pool using `__thread` local storage pointers.
* **Lock-Free Allocation:** Threads allocate from their own pool without acquiring a lock.
* **Fallback:** If a private pool is exhausted, the thread reverts to the global shared heap using the global mutex.



---

## 🧪 Testing & Verification

### Correctness Check

All versions must produce the identical "Final signature."

```bash
./sharedhash bigfile.bin -t > base.out
./esharedhash bigfile.bin -t > exp.out
diff base.out exp.out

```

### Performance Benchmarking

Measure the reduction in lock contention by comparing execution times:

```bash
time ./sharedhash large_input.bin -t
time ./esharedhash large_input.bin -t

```

---

## 📝 Part 1: Threaded Conversion Summary

* **Thread Management:** Replaced `fork()`/`wait()` with `pthread_create()`/`pthread_join()`.
* **Data Flow:** Replaced pipes with a shared `unsigned long results[]` array.
* **Synchronization:** Implemented `pthread_mutex_lock` to ensure allocator consistency.

## 🚀 Part 2: Per-Thread Pool Report

* **Optimization Strategy:** Divided the `UMEM_SIZE` region into equal segments based on thread ID. Each thread manages a "bump pointer" within its segment.
* **Result:** Eliminated the "choke point" of the global mutex for  of allocations.
* **Observation:** In the baseline version, `lock_count` equaled the total number of blocks. In the optimized version, `lock_count` dropped significantly, representing only the initial setup and overflow cases.
* **Performance Gain:** Observed a [X.X]x speedup on files larger than 1MB due to decreased context switching and thread blocking.

---

## 🛠 Debugging & Memory Safety

To check for memory corruption or leaks in your custom allocator:

```bash
valgrind --leak-check=full ./esharedhash sample.txt -t

```

*Note: Ensure the hash signature remains consistent even when pools are exhausted.*

---
