# HPC-JOURNEY: HARDWARE-AWARE COMPUTING & LOW-LEVEL AI SYSTEMS
====================================================================================================
Language:        C++20 | ISO/IEC 14882:2020
Target ISA:      x86-64-v3 (AVX2, FMA3, BMI2)
Parallel Engine: OpenMP 4.5+ (POSIX Threads Backend)
Environment:     Linux x86_64 (Bare-Metal / Ubuntu 22.04 LTS / WSL2)
Toolchain:       GCC 11+ / Clang 14+ (-O3 -march=native -fopenmp)
Author:          Le Cao Minh (HPC & AI Systems Engineering)
Status:          Active Research & Engineering Artifacts
====================================================================================================

> A rigorous systems engineering repository dedicated to the micro-architectural optimization
> of compute-intensive kernels, analytical roofline modeling, cache-subsystem residency, and
> shared-memory parallel runtimes built from first hardware principles.


----------------------------------------------------------------------------------------------------
1. ARCHITECTURAL BLUEPRINT & SUB-SYSTEM CO-DESIGN
----------------------------------------------------------------------------------------------------

 +------------------------------------------------------------------------------------------------+
 |                                    COMPUTATIONAL WORKLOAD                                      |
 |            Dense Linear Algebra (GEMM)  |  IO-Aware Attention  |  Stencil Operators            |
 +------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
 +------------------------------------------------------------------------------------------------+
 |                                  ALGORITHMIC MEMORY ENGINE                                     |
 |   * 2-Level Panel Packing: PackA (Mc x Kc -> L2), PackB (Kc x Nc -> L3/DRAM)                   |
 |   * Cache Blocking: L1 Data Cache (32 KB), L2 Unified (512 KB), L3 Shared (16 MB)              |
 |   * Data Alignment: 64-Byte Cache Line Boundaries (`alignas(64)`)                             |
 +------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
 +------------------------------------------------------------------------------------------------+
 |                                 INSTRUCTION-LEVEL PARALLELISM                                  |
 |   * Explicit Vectorization: 256-bit AVX2 Intrinsics (`__m256`, 8-wide FP32)                   |
 |   * Vector Compute Pipe: Fused Multiply-Add (FMA3: `_mm256_fmadd_ps`, Latency=4, Throughput=0.5)|
 |   * Register Allocation: 6x16 Micro-Kernel pinning 12/16 `%ymm` registers as accumulators      |
 |   * ILP Optimization: 4-way loop unrolling (`K_UNROLL = 4`) hiding pipeline dependency stalls  |
 +------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
 +------------------------------------------------------------------------------------------------+
 |                                    SHARED-MEMORY RUNTIME                                       |
 |   * Hardware Thread Pinning: `OMP_PLACES=cores`, `OMP_PROC_BIND=close` (Anti-Migration)        |
 |   * Coherence Management: Elimination of false sharing / MESI cross-core invalidation buses   |
 |   * Load Distribution: Analytical comparison of `static`, `dynamic(chunk)`, and `guided`       |
 |   * NUMA Topology: Verification of First-Touch page allocation on UMA vs Multi-Socket domains  |
 +------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
 +------------------------------------------------------------------------------------------------+
 |                                  PHYSICAL SILICON SUBSTRATE                                    |
 |          AMD Zen Monolithic APU | 8 Cores / 16 Threads | Dual-Channel DDR4 @ 3200 MT/s         |
 +------------------------------------------------------------------------------------------------+


----------------------------------------------------------------------------------------------------
2. HOST HARDWARE TESTBED & MICRO-ARCHITECTURAL LIMITS
----------------------------------------------------------------------------------------------------

Empirical benchmarks, bandwidth saturation curves, and operational bounds are mapped directly
against the physical specifications of the test execution node:

  Parameter                    Hardware Value                  Micro-architectural Significance
  -------------------------    ----------------------------    ---------------------------------------
  Micro-Architecture           AMD Zen Monolithic APU          Unified single-die core complex (CCX)
  Topology                     8 Cores / 16 Threads (1 Socket) NUMA Nodes: 1 (UMA Monolithic Substrate)
  Instruction Extensions       AVX2, FMA3, BMI2, CLFLUSHOPT    256-bit vector registers (%ymm0 - %ymm15)
  Theoretical Peak (FP32)      1,553.60 GFLOPS                 8 cores * 2 FMA units * 8 FP32 * 4.8 GHz
  L1 Data Cache (Private)      32 KB / core, 8-way assoc.      64-byte line, ~0.9 ns latency (~1.8 TB/s)
  L2 Cache (Private)           512 KB / core, 8-way assoc.     Unified instruction/data, ~3.0 ns latency
  L3 Cache (Shared)            16 MB monolithic, 16-way        Victim cache architecture across 8 cores
  DRAM Physical Subsystem      DDR4-3200 MT/s Dual-Channel     128-bit bus width, theoretical 51.2 GB/s
  Host OS / Kernel             Linux 6.x x86_64                Sub-millisecond high-resolution timers


----------------------------------------------------------------------------------------------------
3. REPOSITORY DIRECTORY SCHEMA & MILESTONE ROADMAP
----------------------------------------------------------------------------------------------------

  Module Subdirectory         Technical Domain              Core Implementations             Status
  -----------------------     ---------------------------   ------------------------------   ---------
  01-matrix-gemm-cpu/         Compute Optimization & SIMD   BLIS 6x16, Packing, FMA3, OMP    COMPLETED
  02-memory-roofline/         Profiling & Empirical Limits  STREAM Triad, CARM Formulation   COMPLETED
  03-multithread-openmp/      Memory Subsystem & Coherence  False Sharing, UMA, Schedulers   COMPLETED
  04-gpu-cuda-basics/         Massively Parallel SIMT       Shared Memory, Coalescing, GEMM  PLANNED
  05-distributed-mpi/         Cluster & Distributed Scale   Non-blocking Halo, Ghost Cell    PLANNED


----------------------------------------------------------------------------------------------------
4. EXPERIMENTAL CASE STUDIES & QUANTITATIVE RESULTS
----------------------------------------------------------------------------------------------------

### CASE STUDY 01: PROGRESSION OF SINGLE-PRECISION GEMM (FP32, N = 2016)
Workload Complexity: 2 * N^3 = 1.638 x 10^10 FLOPs (16.38 GFLOPs)

  Implementation Step             Latency    Throughput      Speedup   Arithmetic Intensity   Regime
  ----------------------------   ---------   ------------   --------   --------------------   --------------
  [01] Naive (i-j-k scalar)      1320.9 ms     12.4 GFLOPS      1.0x         0.25 FLOPs/B     DRAM Bound
  [02] Tiled GEMM (L2/L3 cache)   148.9 ms    110.0 GFLOPS      8.9x         4.00 FLOPs/B     L3 Latency
  [03] AVX2 Naive (Vectorized)     66.8 ms    245.0 GFLOPS     19.8x        16.00 FLOPs/B     L2 Bandwidth
  [04] BLIS 6x16 Micro-Kernel      32.08 ms   510.8 GFLOPS     41.2x       ~336.00 FLOPs/B     COMPUTE BOUND

  Execution Scaling Analysis:
  - Single-Core Vectorized BLIS:  71.00 GFLOPS
  - 8-Core OpenMP Parallel BLIS: 510.76 GFLOPS
  - Multi-Core Scaling Efficiency: 89.92% (Linear scaling against memory-wall constraints)


### CASE STUDY 02: OPENMP ARCHITECTURAL MICRO-BENCHMARKS (`03-multithread-openmp/`)

  [A] False Sharing & Hardware Cache Invalidation (`01_false_sharing.cpp`)
  - Target: 8 threads concurrently updating contiguous 64-bit accumulators over 500M iterations.
  - Baseline (No Padding):       0.4986 ms  --> Extreme MESI invalidation bus traffic / core stalls.
  - Optimized (`alignas(64)`):   0.0049 ms  --> Complete L1D cache-line isolation per physical core.
  - Speedup Delta:               101.57x Performance Recovery.

  [B] First-Touch Memory Allocation Policy (`02_numa_first_touch.cpp`)
  - Target: Streaming read-write transformation across a 512 MB single-precision array.
  - Serial Page Initialization:   365.41 ms
  - Parallel Page Initialization: 366.85 ms
  - Relative Variance:           -0.39% (Statistically indistinguishable).
  - Architectural Takeaway: Confirms hardware substrate is a unified UMA architecture with
    symmetric interconnect latency across all cores to the memory controller.

  [C] Workload Imbalance & Runtime Dispatch Overhead (`03_scheduling_overhead.cpp`)
  - Target: Quadratically skewed execution loop O(i^2) across 20,000 tasks.
  - `schedule(static)`:          0.2223 ms  --> Thread 0-6 stall at barrier waiting for Thread 7.
  - `schedule(dynamic, 1)`:      0.4825 ms  --> Atomic centralized queue contention thrashing.
  - `schedule(dynamic, 16)`:     0.0395 ms  --> Chunk amortization begins to hide queue latency.
  - `schedule(dynamic, 64)`:     0.0196 ms  --> OPTIMAL: 11.3x faster than static scheduling.


----------------------------------------------------------------------------------------------------
5. REPRODUCTION & BUILD INSTRUCTIONS
----------------------------------------------------------------------------------------------------

Prerequisites:
  - Linux x86_64 host environment (WSL2 or Bare-Metal Ubuntu recommended)
  - GCC 11+ with AVX2/FMA flags supported by host CPU (`flags: avx2 fma`)
  - GNU Make utility

Build & Execution Commands:

  # 1. Clone repository
  $ git clone https://github.com/<your-username>/HPC-Journey.git
  $ cd HPC-Journey

  # 2. Compile and execute compute-bound BLIS GEMM benchmark
  $ g++ -O3 -mavx2 -mfma -fopenmp -march=native \
        01-matrix-gemm-cpu/gemm_refactor.cpp -o gemm_bench
  $ OMP_PLACES=cores OMP_PROC_BIND=close ./gemm_bench

  # 3. Compile and execute memory subsystem validation suite
  $ cd 03-multithread-openmp
  $ make all
  $ ./01_false_sharing
  $ ./02_numa_first_touch
  $ ./03_scheduling_overhead


----------------------------------------------------------------------------------------------------
6. PRIMARY ACADEMIC & TECHNICAL REFERENCES
----------------------------------------------------------------------------------------------------

[1] Computer Systems Architecture:
    Bryant, R. E., & O'Hallaron, D. R.
    "Computer Systems: A Programmer's Perspective (CS:APP)", 3rd Edition.
    Pearson / Carnegie Mellon University (CS 15-213).

[2] Operational Intensity & Cache Limits:
    Williams, S., Waterman, A., & Patterson, D. (2009).
    "Roofline: An Insightful Visual Performance Model for Multicore Architectures".
    Communications of the ACM (CACM), 52(4), 65-76.

[3] Matrix Multiplication Frameworks (BLIS / GotoBLAS):
    Goto, K., & van de Geijn, R. A. (2008).
    "Anatomy of High-Performance Matrix Multiplication".
    ACM Transactions on Mathematical Software (TOMS), 34(3), 1-25.

[4] Modern Memory-Efficient AI Kernels:
    Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Re, C. (2022).
    "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness".
    Advances in Neural Information Processing Systems (NeurIPS 2022).

[5] High-Performance Shared Memory Parallelism:
    van der Pas, R., Stotzer, E., & Terboven, C. (2017).
    "Using OpenMP -- The Next Step: High-Performance Simplicity".
    The MIT Press.

====================================================================================================
End of File
====================================================================================================