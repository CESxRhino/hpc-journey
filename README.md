# HPC-Journey: Hardware-Aware Computing & Low-Level AI Systems

[![C++20](https://img.shields.io/badge/Language-C%2B%2B20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Instruction Set](https://img.shields.io/badge/ISA-x86--64--v3%20%7C%20AVX2%20%2B%20FMA3-orange.svg)]()
[![OpenMP](https://img.shields.io/badge/Threading-OpenMP%204.5%2B-green.svg)](https://www.openmp.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Academic Profile](https://img.shields.io/badge/Focus-HPC%20%26%20AI%20Systems-red.svg)]()

> A systems engineering repository systematically documenting the optimization of dense compute kernels, multi-level memory hierarchy modeling, and hardware-aware parallelism from first principles.

---

## 📌 Architectural Blueprint

===================================================================================
                             HPC & AI SYSTEMS RUNTIME
===================================================================================
 [Algorithm Layer]       GEMM / BLAS-3         FlashAttention       Stencil/PDEs
                               │                      │                   │
 ──────────────────────────────┼──────────────────────┼───────────────────┼────────
 [Optimization Engine]    Panel Packing        Shared Memory       Ghost Exchange
                          Reg Tiling (6x16)    IO-Aware Tiling     Ring Topologies
                               │                      │                   │
 ──────────────────────────────┼──────────────────────┼───────────────────┼────────
 [Execution Model]        AVX2 / FMA3             OpenMP            MPI / NCCL
                          Vectorization        Thread Binding      Interconnect
                               │                      │                   │
 ──────────────────────────────┼──────────────────────┼───────────────────┼────────
 [Hardware Substrate]     L1D/L2 Cache         MESI Coherence      NUMA / PCIe Bus
                          (32KB / 512KB)       (64B Cache Line)    (Dual-Channel)
===================================================================================

---

## 💻 Target System Specifications

All micro-benchmarks and roofline models are profiled on the following host environment:

| Microarchitectural Parameter | Specification | Notes / Constraints |
| :--- | :--- | :--- |
| **Processor Model** | AMD Zen Architecture (8 Cores / 16 Threads) | Base 2.0 GHz, Boost up to 4.5 GHz |
| **Instruction Extensions** | AVX2, FMA3, BMI2, SSE4.2 | 256-bit Vector Width (8-wide FP32) |
| **Theoretical FP32 Peak** | **1,553.6 GFLOPS** | Dual FMA pipes @ Peak Boost |
| **L1 Data Cache** | 32 KB / core, 8-way set associative | 64-byte line, ~0.9 ns latency |
| **L2 Unified Cache** | 512 KB / core, 8-way set associative | ~3.0 ns latency |
| **L3 Shared Cache** | 16 MB monolithic, 16-way set associative | Shared across 8 cores within CCX |
| **DRAM Subsystem** | DDR4 Dual-Channel @ 3200 MT/s | Peak Bandwidth ~51.2 GB/s |
| **OS & Toolchain** | Ubuntu Linux x86_64, GCC 11+ (`-O3 -march=native`) | Clang/LLVM validation available |

---

## 🔬 Repository Structure & Milestone Progress

.
├── 01-matrix-gemm-cpu/          # BLIS-like GEMM micro-kernel, packing, AVX2 FMA
├── 02-memory-roofline/          # Cache-Aware Roofline Model (CARM), STREAM benchmark
├── 03-multithread-openmp/       # False sharing, NUMA first-touch, workload scheduling
├── 04-gpu-cuda-basics/          # [Planned] CUDA GEMM, shared memory tiling, coalescing
└── 05-distributed-mpi/          # [Planned] Domain decomposition, non-blocking MPI

| Sub-System | Domain | Optimization Primitives Demonstrated | Status |
| :--- | :--- | :--- | :---: |
| **`01-matrix-gemm-cpu`** | Compute Core | 2-level panel packing, 6x16 register blocking, FMA unrolling | `Completed` |
| **`02-memory-roofline`** | Limits Analysis | STREAM triad, cache-aware operational intensity, CARM bounds | `Completed` |
| **`03-multithread-openmp`** | Memory Subsystem | Cache line padding (`alignas(64)`), UMA profiling, dynamic chunks | `Completed` |
| **`04-gpu-cuda-basics`** | SIMT Accelerators | Block tiling, bank conflicts, memory coalescing, warp divergence | `Planned` |
| **`05-distributed-mpi`** | Distributed Scale | Point-to-point non-blocking routines, collective reduction | `Planned` |

---

## 📊 Key Case Studies & Benchmark Metrics

### 1. General Matrix Multiply (GEMM FP32, N = 2016)
Evolutionary progression of a single-precision matrix product (C = A * B + C) from un-vectorized baseline to a compute-bound micro-kernel:

12.4 GFLOPS  ──►  110.0 GFLOPS  ──►  245.0 GFLOPS  ──►  510.8 GFLOPS
[Naive IJK]       [L2/L3 Tiled]       [AVX2 Naive]       [BLIS 6x16 + OpenMP]

| Implementation | Architectural Optimization | Latency | GFLOPS | Arithmetic Intensity | Bottleneck Regime |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `01_naive_ijk` | Stride-N memory access, scalar arithmetic | ~1320 ms | 12.4 | 0.25 FLOP/B | DRAM Bandwidth |
| `02_tiled_gemm` | Sub-matrix blocking fitting L2/L3 cache | ~149 ms | 110.0 | 4.0 FLOP/B | L3 Latency Bound |
| `03_avx2_naive` | Vectorized loop, unaligned strided vector loads | ~67 ms | 245.0 | 16.0 FLOP/B | L2 Bandwidth Bound |
| **`04_blis_6x16`** | **Panel Packing + 12x %ymm Accumulators + OpenMP** | **32.08 ms** | **510.8** | **~336.0 FLOP/B** | **Compute Plateau** |

*Core Takeaway:* By pre-packing matrices A and B into continuous, aligned buffers and keeping intermediate values in 12 hardware vector registers (6 x 16 tile), TLB and cache thrashing were eliminated, resulting in an **89.9% parallel scaling efficiency** across 8 physical cores.

---

### 2. Multi-threaded Architecture Benchmarks (`03-multithread-openmp`)

#### A. False Sharing & MESI Coherence Contention
* **Observation:** When 8 threads concurrently increment contiguous 8-byte counters, the cores repeatedly invalidate each other's L1 Data Cache lines via bus traffic.
* **Resolution:** Applied `alignas(64)` padding to guarantee each counter occupies a dedicated cache line.
* **Result:** Execution latency dropped from **`0.4986 ms`** to **`0.0049 ms`** (**101.57x Speedup**).

#### B. First-Touch Policy on Monolithic APUs
* **Observation:** Evaluating serial vs. parallel first-touch on a 512 MB buffer revealed negligible variance (**`-0.39%`**).
* **Architectural Insight:** Confirms that the host processor functions as an **UMA (Uniform Memory Access)** monolithic die, where all 8 cores access the integrated memory controller (IMC) with symmetric latency.

#### C. Scheduling Overheads under Workload Skew (N = 20,000)
* **Observation:** Evaluating scheduling strategies on quadratic computational load imbalance (O(i^2)):
  * `schedule(static)`: **`0.2223 ms`** (Early-finishing threads stalled at barrier).
  * `schedule(dynamic, 1)`: **`0.4825 ms`** (Devastating atomic queue lock contention).
  * `schedule(dynamic, 64)`: **`0.0196 ms`** (**11.3x speedup vs. static**, balancing load with minimal queue overhead).

---

## 🛠️ Build and Reproduction

### Prerequisites
* GCC 11+ or Clang 14+ with OpenMP and AVX2 support
* GNU Make 4.x
* Linux Environment (x86_64)

### Reproducing Benchmarks

```bash
# 1. Clone repository
git clone [https://github.com/](https://github.com/)<your-username>/HPC-Journey.git
cd HPC-Journey

# 2. Build and run BLIS GEMM micro-kernel
g++ -O3 -mavx2 -mfma -fopenmp -march=native \
    01-matrix-gemm-cpu/gemm_refactor.cpp -o gemm_bench
./gemm_bench

# 3. Build and profile OpenMP architectural suite
cd 03-multithread-openmp
make all
./01_false_sharing
./02_numa_first_touch
./03_scheduling_overhead
```
### Selected References & Foundations
Systems Architecture & Micro-benchmarking

    Bryant, R. E., & O'Hallaron, D. R. Computer Systems: A Programmer's Perspective (CS:APP 15-213). Carnegie Mellon University.

    Williams, S., Waterman, A., & Patterson, D. Roofline: An Insightful Visual Performance Model for Multicore Architectures. CACM, 2009.

    Hennessy, J. L., & Patterson, D. A. Computer Architecture: A Quantitative Approach, 6th Edition.

High-Performance BLAS & Operator Design

    Goto, K., & van de Geijn, R. A. Anatomy of High-Performance Matrix Multiplication. ACM TOMS, 2008.

    Van Zee, F. G., & van de Geijn, R. A. BLIS: A Framework for Rapidly Instantiating BLAS Functionality. ACM TOMS, 2015.

    Dao, T., et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022.