# Performance Report: Part 1 - Multi-Thread Conversion and Contention Analysis

## First Observations

**Small File (1KB):** Both Single-Process and Multi-Process modes are completed successfully and quickly.

**Large File (100KB):** The Multi-Process mode (`-m`) consistently failed with a **Segmentation Fault** or a **fatal memory exhaustion error** (e.g., *umalloc failed for block X*).

---

## Implement Threaded Mode (`-t`)

I have successfully implemented the multi-threaded mode (`-t`) under a single shared memory allocator following steps in the instructions.

### **Thread Management**
Replaced `fork()`/pipes with:
- `pthread_create()`
- `pthread_join()`
- a shared results array

### **Synchronization**
Replaced the global semaphore with a `pthread_mutex` to guard all critical sections in `_umalloc` and `_ufree`.

### **Batching Strategy**
Implemented a batching mechanism (`MAX_CONCURRENT_THREADS = 32` or `64`):
- The parent process is forced to wait for a batch of threads to finish
- Then it spawns the next batch

---

## Performance Bottleneck

However, performance under medium-to-large workloads (100KB to 1MB) revealed a **catastrophic bottleneck**, with parallel execution running **over 12× slower** than the sequential baseline.

This confirms the implementation is **correct but fundamentally unscalable** due to severe **lock contention**.

---

## What I Learned from Part 1

The multi-threaded implementation (`-t`) was successful in achieving correctness and stability under large file workloads.

However, the performance results for larger files are unacceptable:

### **Performance Table**

| Workload | Mode              | Real Time (Wall Clock) | Time Factor         |
|----------|-------------------|--------------------------|----------------------|
| 1KB      | Single-Process    | 0.004s                   | 1× (Baseline)        |
| 1KB      | Multi-Thread (-t) | 0.003s                   | 0.75× Faster         |
| 1MB      | Single-Process    | 0.269s                   | 1× (Baseline)        |
| 1MB      | Multi-Thread (-t) | 3.255s                   | **12.1× Slower**    |

---

## Cause: Global Lock Contention

The fast time for the 1KB file is expected — negligible work means negligible contention.

As the workload scales, the slowdown becomes **exponential**.

### **Why the Slowdown?**

- Huffman encoding is **allocation-heavy**, requiring frequent allocation/free operations.
- With **32 threads** running simultaneously and constantly allocating/freeing nodes, all threads pile up on the **single `pthread_mutex`** that guards the allocator.
- The lock allows only **one thread** into the heap structure at a time.

### **Performance Impact**

- The workload effectively becomes **sequential** at the allocator.
- Threads spend more time **blocked** on the mutex than doing computation.
- The OS spends significant time **context-switching and waking threads**, raising system time.
- Result: **Massive degradation in performance**.

---
