# OS-Managed-Memory
A concurrent memory allocator that explores the transition from multi-process shared memory to multi-threaded execution. This project implements a thread-safe, first-fit heap allocator using pthreads and optimizes performance by implementing Per-Thread Memory Pools to eliminate global lock contention.
