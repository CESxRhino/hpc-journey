# High-Performance Computing (HPC) Journey
A comprehensive repository covering theoretical foundations, hardware-aware optimizations, and parallel programming paradigms in modern High-Performance Computing.
---
## HPC Core Curriculum & Architecture Map
```text
                  +-----------------------------------------------+
                  |            HPC System Architecture            |
                  +-----------------------┬-----------------------+
                                          |
        +---------------------------------+-------------------------------+
        |                                                                 |
+-------v-----------------------+               +-------------------------v-----+
|        Shared Memory &        |               |      Distributed Systems &    |
|       Node Architecture       |               |        Cluster Computing      |
+-------------------------------+               +-------------------------------+
| * Cache & Memory Hierarchy    |               | * Message Passing (MPI)       |
| * Instruction-Level (ILP/SIMD)|               | * Interconnects (InfiniBand)  |
| * Multithreading (OpenMP)     |               | * Partitioning & Load Balance |
| * Hardware Accelerators (GPU) |               | * Distributed Storage/I/O     |
+-------------------------------+               +-------------------------------+
```
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
| `02-memory-roofline/` | Profiling & Limits | STREAM Benchmark, Hardware Prefetching, Empirical Roofline Analysis | Completed |
| `03-multithread-openmp/` | Shared Memory | False Sharing Detection, NUMA First-Touch, Dynamic Scheduling | In Progress |
| `04-gpu-cuda-basics/` | Massively Parallel | Thread/Block Grids, Shared Memory Tiling, Memory Coalescing | Planned |
| `05-distributed-mpi/` | Distributed Memory | Ghost Cell Exchange, Ring Topology Communication, Stencil Computation | Planned |

---

## 3. Case Study 01: Deep-Dive CPU GEMM Optimization
An empirical progression optimizing single-precision General Matrix Multiplication (GEMM, $C = A \times B + C$) across 5 evolutionary stages:

```text
01_naiveijk (~3.4 GFLOPS)
   │   [Linear Stride-1 traversal]
   ▼
02_optikj (~42.9 GFLOPS)
   │   [L1/L2 Cache Tiling sub-blocks]
   ▼
03_cachetliing (~24.7 GFLOPS - Single-Core scalar/auto-vectorized)
   │   [AVX2 Micro-Kernel (4x8) + OpenMP Multi-threading]
   ▼
04_micro_kenel (> 170 GFLOPS)
   │   [Continuous Panel Packing + Unroll by 4 + Prefetching]
   ▼
05_blis_packed_complete (~198 GFLOPS / ~10.8 ms @ N=1024)
```
### Benchmark Summary (Single-Precision `float`)

| Implementation File | Core Optimization Techniques | Time ($N=1024$) | Throughput (GFLOPS) | Speedup vs Naive |
| :--- | :--- | :---: | :---: | :---: |
| **`01_naiveijk.cpp`** | Sequential $i-j-k$ loops, Stride-$N$ memory access, scalar | ~625 ms | ~3.4 | 1.0x (Baseline) |
| **`02_optikj.cpp`** | Loop reordering to $i-k-j$, continuous Stride-1 traversal | ~50 ms | ~42.9 | ~12.5x |
| **`03_cachetliing.cpp`** | $64 \times 64$ sub-blocks fitting into L1/L2 Cache (Single-core) | ~87 ms | ~24.7 | ~7.2x |
| **`04_micro_kenel.cpp`** | $4 \times 8$ AVX2 micro-kernel + FMA + OpenMP parallelization | ~12.6 ms | ~170.5 | ~50.1x |
| **`05_blis_packed_complete.cpp`** | Panel Packing + Unroll by 4 + `_mm_prefetch` + OpenMP | **~10.8 ms** | **~197.7** | **~58.0x** |
---
### Microarchitectural Validation (`perf stat` on `05_blis_packed_complete`)

Profiling metrics captured via Linux hardware performance counters for $N = 1024$:
* **Theoretical Workload:** $2 \times N^3 = 2 \times 1024^3 \approx 2.147 \times 10^9$ FLOPs ($2.15\text{ GFLOPs}$).
* **Execution Efficiency:** $\approx 396.25 \times 10^6$ CPU instructions executed ($\approx 5.42\text{ FLOPs/instruction}$), confirming high density of 256-bit FMA vector instructions (`vfmadd231ps`).
* **Cache Miss Mitigation:** Only $\approx 2.56 \times 10^6$ LLC cache misses and $\approx 10.47 \times 10^6$ L1-dcache misses, validating the effectiveness of continuous panel packing and software prefetching.
* **Multithreading Scaling:** Wall-clock time of $0.022\text{ s}$ against $0.169\text{ s}$ user time demonstrates an effective $\approx 7.6\times$ parallel scaling efficiency across available CPU threads.
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