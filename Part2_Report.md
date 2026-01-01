# Performance Report: Part 2 - Global Lock Elimination (Strategy A)

## Goal and Strategy

**Goal:**  
Eliminate the severe global lock contention observed in Part 1, where the multi-threaded mode (`-t`) ran **≈12.1× slower** than the single-process baseline due to serialization inside the memory allocator.

**Strategy A: Per-Thread Memory Pools**  
This stratergy was chosen because it directly addresses the contention,reduced lock time, improved parallelism, lower system time, easy to implement and debug. By reserving a **32 KB** local memory chunk for each thread, memory allocation/deallocation shifts from a slow, globally locked operation to a **fast, thread-local, lock-free bump-pointer allocator** for most calls.

---

## Performance Comparison between Multi-Thread mode from Part 1 and the Improved Version 

The results show a complete reversal of the performance issues observed in Part 1, now demonstrating **massive speedup through effective parallelism**.

### **Performance Table**

| Workload | Mode | Part | Real Time (Wall Clock) | Time Factor (vs Baseline) |
|---------|------|------|--------------------------|----------------------------|
| 1 MB | Single-Process | Baseline | **0.272 s** | **1.0×** |
| 1 MB | Multi-Thread (`-t`) | Part 1 (Global Lock) | **3.416 s** | **12.56× Slower** |
| 1 MB | Multi-Thread (`-t`) | Part 2 (Thread Pools) | **0.051 s** | **5.33× Faster** |

---

## Key Finding

The Per-Thread Memory Pool implementation made the multi-threaded processing of the **1 MB** file:

- **5.33× faster** than the sequential baseline  
- **~67× faster** than the previous global-lock implementation  

---

## Analysis of Concurrency Success

The performance data clearly shows that the global lock was completely eliminated as the primary bottleneck.

### CPU Time Breakdown

| Metric | Part 1 (Global Lock) | Part 2 (Thread Pools) | Interpretation |
|--------|------------------------|-------------------------|----------------|
| **Real Time** | 3.416 s | **0.051 s** | User waits far less time |
| **User Time** | ≈0.264 s | 0.090 s | Faster execution due to reduced allocation overhead |
| **System Time** | ≈3.100 s | 0.059 s | Lock/schedule overhead drastically reduced |
| **Total CPU Time** | ≈3.364 s | **0.149 s** | Total work done across all cores |

---

## Proof of Parallelism

The strongest evidence of true parallel execution: Real Time (0.051s) < Total CPU Time (0.149s)

Because the CPU spent **0.149 s** across all cores, yet the wall-clock time was only **0.051 s**, multiple threads were clearly running **simultaneously** across different processors.

This shows that the workload was successfully **parallelized**, and execution time was significantly compressed.

---

## Conclusion

**Strategy A (Per-Thread Memory Pools) was highly effective.**

By giving each thread a **lock-free memory sandbox**, the memory-intensive Huffman encoding workload transitioned from a **globally serialized bottleneck** to a **fully parallel, scalable design**. The implemented optimization (Strategy A) dramatically improved concurrency by introducing **Thread-Local Storage (TLS)** memory pools for the multi-threaded (-t) execution. This created a **lock-free fast path (bump allocation)** where the majority of memory requests are handled independently within a **thread's private pool**, minimizing reliance on the global free list and its mutex (mLock_thread). This demonstrated an important point in scaling: more threads do not equate to more speed if they frequently fight over shared resources; true performance gains come from fine-grained, concurrent designs that prioritize memory locality and reduce contention.

