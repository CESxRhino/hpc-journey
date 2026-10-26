# High-Performance Computing (HPC) Journey
A comprehensive repository covering theoretical foundations, hardware-aware optimizations, and parallel programming paradigms in modern High-Performance Computing.
---
## HPC Core Curriculum & Architecture Map

              +----------------------------------------------+
              |           HPC System Architecture            |
              +----------------------┬-----------------------+
                                     |
    +--------------------------------┴-------------------------------+
    |                                                                |
+-------v-----------------------+               +------------------------v------+
|       Shared Memory &         |               |     Distributed Systems &     |
|       Node Architecture       |               |       Cluster Computing       |
+-------------------------------+               +-------------------------------+
| * Cache & Memory Hierarchy    |               | * Message Passing (MPI)       |
| * Instruction-Level (ILP/SIMD)|               | * Interconnects (InfiniBand)  |
| * Multithreading (OpenMP)     |               | * Partitioning & Load Balance |
| * Hardware Accelerators (GPU) |               | * Distributed Storage/I/O     |
+-------------------------------+               +-------------------------------+


---

## 1. Pillars of High-Performance Computing

### Pillar 1: Hardware-Aware Computing & Memory Hierarchy
* **Roofline Model:** Quantifies whether computational kernels are compute-bound or memory-bandwidth bound relative to hardware operational intensity limits.
* **Cache Locality:** Exploits spatial and temporal locality to eliminate compulsory, capacity, and conflict cache misses.
* **Cache Tiling / Blocking:** Partitions large tensors into L1/L2 cache-resident sub-blocks to minimize high-latency DRAM round-trips.
* **Memory Alignment & Padding:** Aligns memory addresses to 32-byte or 64-byte vector boundaries to maximize bus throughput and enable aligned SIMD instructions.

### Pillar 2: Instruction-Level Parallelism (ILP) & Vectorization (SIMD)
* **Pipelining & Superscalar Execution:** Minimizes data dependencies, hazard stalls, and branch mispredictions.
* **SIMD Intrinsics (AVX2 / AVX-512):** Explicit register-level vector processing operating simultaneously on 8 single-precision floats (256-bit) or 16 floats (512-bit).
* **Fused Multiply-Add (FMA):** Executes single-cycle vector multiplication and accumulation (`A * B + C`).
* **Register Tiling:** Carefully manages hardware vector register allocation (YMM/ZMM) to eliminate stack spilling.

### Pillar 3: Shared-Memory Parallelism (OpenMP & Multithreading)
* **Thread Scheduling:** Evaluates static, dynamic, and guided work-sharing strategies across symmetric multiprocessing (SMP) cores.
* **Synchronization & Overhead:** Analyzes barrier synchronization costs, critical sections, and atomic operation primitives.
* **False Sharing Mitigation:** Prevents cross-core cache invalidation thrashing by isolating thread writes across distinct 64-byte cache lines.
* **NUMA Awareness:** Mitigates non-uniform memory access latency on multi-socket architectures via first-touch memory allocation policies.

### Pillar 4: Distributed-Memory Computing & Accelerators
* **Message Passing Interface (MPI):** Point-to-point non-blocking communications (`MPI_Isend`, `MPI_Irecv`) and collective operations (`MPI_Bcast`, `MPI_Reduce`, `MPI_Allreduce`).
* **Heterogeneous Computing:** Massively parallel data throughput modeling (SIMT) using NVIDIA CUDA and OpenCL.
* **Parallel I/O & Distributed Storage:** High-throughput data streaming utilizing MPI-IO, distributed file systems (Lustre, GPFS), and distributed object stores/Data Lakes.

---

## 2. Benchmark Experiments & Artifacts

| Directory | Topic / Paradigm | Key Techniques Demonstrated | Status |
| :--- | :--- | :--- | :---: |
| `01-matrix-gemm-cpu/` | Memory & SIMD | Loop Reordering, Cache Tiling, AVX2 FMA Micro-Kernel, OpenMP | Completed |
| `02-memory-roofline/` | Profiling & Limits | STREAM Benchmark, Hardware Prefetching, Empirical Roofline Analysis | In Progress |
| `03-multithread-openmp/` | Shared Memory | False Sharing Detection, NUMA First-Touch, Dynamic Scheduling | Planned |
| `04-gpu-cuda-basics/` | Massively Parallel | Thread/Block Grids, Shared Memory Tiling, Memory Coalescing | Planned |
| `05-distributed-mpi/` | Distributed Memory | Ghost Cell Exchange, Ring Topology Communication, Stencil Computation | Planned |

---

## 3. Case Study 01: Deep-Dive CPU GEMM Optimization

An empirical progression optimizing single-precision General Matrix Multiplication (GEMM, $C = A \times B + C$) with matrix dimensions $M = N = K = 2048$:

01_naive_ijk (~3.4 GFLOPS)
|   [Linear Stride-1 access traversal]
v
02_loop_ikj (~42.9 GFLOPS)
|   [L1/L2 Cache Tiling sub-blocks]
v
03_cache_tiling (~24.7 GFLOPS - Single-Core scalar/auto-vectorized)
|   [AVX2 Micro-Kernel (4x8) + Memory Packing + OpenMP Multi-Core]
v
04_blis_avx2_openmp (> 170 GFLOPS)


### Benchmark Summary ($N = 2048$, Single-Precision `float`)

| Implementation | Execution Time (ms) | Throughput (GFLOPS) | Speedup vs Naive |
| :--- | :---: | :---: | :---: |
| **01. Naive GEMM ($i-j-k$, `-O3`)** | ~5,000 ms | ~3.4 | 1.0x (Baseline) |
| **02. Loop Reorder ($i-k-j$, `-O3`)** | ~400 ms | ~42.9 | ~12.5x |
| **03. Cache Tiling + AVX2 (Single-core)** | ~695 ms | ~24.7 | ~7.2x |
| **04. Micro-Kernel + AVX2 + OpenMP** | **< 100 ms** | **> 170** | **> 50x** |

* **Theoretical FLOP count:** $2 \times M \times N \times K \approx 17.18 \times 10^9$ operations for $N = 2048$.
* **Performance Profiling:** Disassembly validation (`objdump` / `g++ -S`), Hardware Performance Counters (`perf stat`).

---

## 4. Build and Execution

### Prerequisites
* GCC/G++ version 11+ with AVX2, FMA, and OpenMP runtime support
* Linux bare-metal or WSL2 environment (x86-64-v3 target)

### Compilation
```bash
# Build the optimized BLIS-like GEMM implementation
g++ -O3 -mavx2 -mfma -fopenmp -march=native gemm_blis.cpp -o gemm_blis

# Execute benchmark
./gemm_blis
5. Primary References
CS:APP (CMU 15-213): Computer Systems: A Programmer's Perspective – Randal E. Bryant & David R. O'Hallaron.

Stanford CS149: Parallel Computing – Kayvon Fatahalian.

GotoBLAS / BLIS Framework: Anatomy of High-Performance Matrix Multiplication – Kazushige Goto & Robert A. van de Geijn (ACM TOMS).