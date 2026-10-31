# HPC-Journey: Hardware-Aware Computing & Low-Level AI Systems

> **Systematic exploration of micro-architectural optimizations, analytical roofline modeling, and shared-memory parallel runtimes on modern multicore architectures.**

---

### Highlights & Empirical Breakthroughs

* **Compute Saturation:** Achieved **510.8 GFLOPS** (~32.9% of Theoretical Peak) on FP32 GEMM ($N=2016$) with an **89.9% parallel scaling efficiency** across 8 physical cores.
* **Cache Ping-Pong Elimination:** Completely eliminated False Sharing using `alignas(64)`, demonstrating a **101.57x speedup** over the unaligned structure.
* **Workload Auto-Balancing:** Mitigated quadratic workload skew barrier stalls via `dynamic(64)` scheduling, outperforming `static` scheduling by **11.3x**.
* **Hardware Limit Profiling:** Profiled multi-level cache and DRAM bandwidth ceilings to construct empirical Cache-Aware Roofline Models (CARM).

---

## Architecture Blueprint

+---------------------------------------------------------------------------------------+
|                                COMPUTATIONAL WORKLOAD                                 |
|          Dense Linear Algebra (GEMM)  |  IO-Aware Tiling  |  Stencil Operators        |
+---------------------------------------------------------------------------------------+
                                           │
                                           ▼
+---------------------------------------------------------------------------------------+
|                                ALGORITHMIC MEMORY ENGINE                              |
|   * 2-Level Panel Packing : PackA (Mc x Kc -> L2), PackB (Kc x Nc -> L3/DRAM)         |
|   * Cache Blocking        : L1D (32 KB), L2 (512 KB), L3 Shared (16 MB)               |
|   * Alignment & Padding   : 64-byte cache-line isolation (`alignas(64)`)              |
+---------------------------------------------------------------------------------------+
                                           │
                                           ▼
+---------------------------------------------------------------------------------------+
|                             INSTRUCTION-LEVEL PARALLELISM                             |
|   * Explicit Vectorization: 256-bit AVX2 Intrinsics (`__m256`, 8-wide FP32)           |
|   * Pipeline Density      : Fused Multiply-Add (FMA3: `_mm256_fmadd_ps`)              |
|   * Register Allocation   : 6x16 tile pinning 12/16 `%ymm` registers                  |
|   * ILP Optimization      : 4-way loop unrolling to saturate execution ports          |
+---------------------------------------------------------------------------------------+
                                           │
                                           ▼
+---------------------------------------------------------------------------------------+
|                                 SHARED-MEMORY RUNTIME                                 |
|   * Thread Pinning        : `OMP_PLACES=cores`, `OMP_PROC_BIND=close`                 |
|   * Work Scheduling       : Dynamic chunk dispatch against quadratic workload skew    |
|   * Topology Verification : UMA monolithic validation on unified memory controller    |
+---------------------------------------------------------------------------------------+

---

## Testbed Specifications

| Subsystem | Physical Specification | Microarchitectural Role |
| :--- | :--- | :--- |
| **Processor** | AMD Zen Architecture (8C / 16T) | Unified CCX, monolithic die |
| **Vector Engine** | AVX2 + FMA3 (256-bit wide) | Dual FMA execution ports |
| **FP32 Peak** | **1,553.6 GFLOPS** | Upper-bound theoretical ceiling |
| **L1D Cache** | 32 KB / core, 8-way assoc | ~0.9 ns latency, 64-byte lines |
| **L2 Cache** | 512 KB / core, 8-way assoc | ~3.0 ns latency |
| **L3 Cache** | 16 MB shared across 8 cores | Victim cache structure |
| **DRAM** | DDR4-3200 MT/s Dual-Channel | ~51.2 GB/s theoretical throughput |

---

## Benchmark Results & Case Studies

### 1. General Matrix Multiplication (`01-matrix-gemm-cpu/`, FP32, N = 2016)
Workload complexity: $2 \times N^3 \approx 16.38\text{ GFLOPs}$.

| Implementation Source File | Core Optimization Primitives | Latency | Throughput | Speedup | Regime |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `01_naiveijk.cpp` | Stride-$N$ memory access, scalar arithmetic | 1320.9 ms | 12.4 GFLOPS | 1.0x | DRAM Bound |
| `02_optikj.cpp` | Loop interchange ($i$-$k$-$j$) for sequential reads | 425.2 ms | 38.5 GFLOPS | 3.1x | Cache-Miss Reduction |
| `03_cachetilling.cpp` | Hierarchical 3-level cache blocking ($M_c, N_c, K_c$) | 148.9 ms | 110.0 GFLOPS | 8.9x | L3 Latency Bound |
| `04_micro_kenel.cpp` | AVX2 + FMA3 vectorization, unaligned loads | 66.8 ms | 245.0 GFLOPS | 19.8x | L2 Bandwidth Bound |
| **`05_blis_packed_complete.cpp`** | **2-Level Packing + 6x16 Register Tiling + OpenMP** | **32.08 ms** | **510.8 GFLOPS** | **41.2x** | **Compute Plateau** |

> **Analysis:** Partitioning the computation into an optimized $6 \times 16$ register tile retains intermediate accumulations across 12 dedicated `%ymm` registers. Coupled with 2-level contiguous buffer packing ($A$ panel to L2, $B$ panel to L3), this approach eliminates capacity/conflict cache misses and TLB paging stalls, driving throughput directly into the compute-bound plateau.

---

### 2. Empirical Cache-Aware Roofline Model (`02-memory-roofline/`)
Using `benchmark_roofline.cpp` and `plot.py`, multi-level memory throughput is empirically characterized across cache and main memory boundaries.

| Memory Hierarchy Level | Sustained Bandwidth | Latency | Operational Intensity Ridge Point |
| :--- | :---: | :---: | :---: |
| **L1 Data Cache** | ~1,850 GB/s | ~0.9 ns | ~0.84 FLOP/Byte |
| **L2 Unified Cache** | ~820 GB/s | ~3.0 ns | ~1.89 FLOP/Byte |
| **L3 Shared Cache** | ~380 GB/s | ~10.5 ns | ~4.08 FLOP/Byte |
| **DRAM Subsystem** | ~38.4 GB/s | ~75.0 ns | ~40.45 FLOP/Byte |

<p align="center">
  <img src="02-memory-roofline/carm_roofline.png" alt="Cache-Aware Roofline Model" width="850">
</p>

> **Roofline Analysis:** Kernel execution with arithmetic intensity below **~40.5 FLOPs/Byte** is strictly bounded by DRAM bandwidth. By adopting register tiling ($6 \times 16$) and 2-level contiguous buffer packing, GEMM operational intensity jumps from **0.25 FLOP/Byte** to **~336.0 FLOP/Byte**, shifting the workload into the compute-bound plateau.

---

### 3. OpenMP Architectural Suite (`03-multithread-openmp/`)

#### False Sharing Elimination (`01_false_sharing.cpp`)
* **Problem:** 8 threads concurrently incrementing contiguous 8-byte scalar variables located within the same 64-byte cache line, causing relentless inter-core MESI bus invalidation traffic.
* **Fix:** Applied 64-byte boundary padding (`alignas(64)`) to isolate thread-local state onto distinct cache lines.
* **Latency:** Dropped from `0.4986 ms` down to `0.0049 ms` (**Speedup: 101.57x**).

#### First-Touch Policy Profiling (`02_numa_first_touch.cpp`)
* **Observation:** Serial versus parallel first-touch buffer initialization across a 512 MB memory buffer revealed a negligible variance of `-0.39%`.
* **Architectural Insight:** Confirms the target host platform operates as an authentic Uniform Memory Access (UMA) monolithic die architecture (single NUMA node), where all physical cores experience uniform latency to the integrated memory controller.

#### Workload Balancing (`03_scheduling_overhead.cpp`)
* **Observation:** Evaluated under severe quadratic workload imbalance ($\mathcal{O}(i^2)$ across 20,000 tasks):
  * `static`: `0.2223 ms` (Cores assigned lighter early iterations stall idle at the barrier awaiting completion of final iterations).
  * `dynamic, 1`: `0.4825 ms` (Severe atomic lock queue contention due to task dispatch granularity being too fine).
  * `dynamic, 64`: **`0.0196 ms`** (**11.3x faster than static**, optimally amortizing synchronization overhead while maintaining workload balance).

---

## Quickstart & Reproducibility

$ git clone https://github.com/<your-username>/HPC-Journey.git
$ cd HPC-Journey

# 1. Build and run CPU GEMM optimization stages
$ cd 01-matrix-gemm-cpu
$ g++ -O3 -mavx2 -mfma -fopenmp -march=native 05_blis_packed_complete.cpp -o blis_bench
$ OMP_PLACES=cores OMP_PROC_BIND=close ./blis_bench

# 2. Profile empirical Roofline model and generate plots
$ cd ../02-memory-roofline
$ g++ -O3 -mavx2 -fopenmp -march=native benchmark_roofline.cpp -o benchmark_roofline
$ ./benchmark_roofline
$ python3 plot.py

# 3. Build and execute OpenMP micro-architectural suite
$ cd ../03-multithread-openmp && make all
$ ./01_false_sharing
$ ./02_numa_first_touch
$ ./03_scheduling_overhead

---

## References

### 1. Computer Systems Architecture & Performance Modeling
[1] Bryant, R. E., & O'Hallaron, D. R. (2016).
    Computer Systems: A Programmer's Perspective (CS:APP 15-213), 3rd Edition.
    Pearson / Carnegie Mellon University.

[2] Williams, S., Waterman, A., & Patterson, D. (2009).
    Roofline: An Insightful Visual Performance Model for Multicore Architectures.
    Communications of the ACM (CACM), 52(4), 65–76.

[3] Hennessy, J. L., & Patterson, D. A. (2019).
    Computer Architecture: A Quantitative Approach, 6th Edition.
    Morgan Kaufmann / Elsevier.

### 2. High-Performance Linear Algebra & BLIS Framework
[4] Goto, K., & van de Geijn, R. A. (2008).
    Anatomy of High-Performance Matrix Multiplication.
    ACM Transactions on Mathematical Software (TOMS), 34(3), 1–25.

[5] Van Zee, F. G., & van de Geijn, R. A. (2015).
    BLIS: A Framework for Rapidly Instantiating BLAS Functionality.
    ACM Transactions on Mathematical Software (TOMS), 41(3), 1–33.

[6] Low, T. M., Igual, F. D., Smith, T. M., & Quintana-Ortí, E. S. (2016).
    Analytical Modeling Is Enough for High-Performance BLIS.
    ACM Transactions on Mathematical Software (TOMS), 43(2), 1–18.

### 3. Shared-Memory Parallelism & OpenMP Runtime
[7] van der Pas, R., Stotzer, E., & Terboven, C. (2017).
    Using OpenMP -- The Next Step: High-Performance Simplicity.
    The MIT Press.

[8] Chandra, R., Dagum, L., Kohr, D., Maydan, D., McDonald, J., & Menon, R. (2001).
    Parallel Programming in OpenMP.
    Morgan Kaufmann Publishers.

### 4. Modern IO-Aware AI Kernels & Serving Infrastructures
[9] Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022).
    FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.
    Advances in Neural Information Processing Systems (NeurIPS 2022).

[10] Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., ... & Stoica, I. (2023).
     Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM).
     Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP 2023).