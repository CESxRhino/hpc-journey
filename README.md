# HPC-Journey: Hardware-Aware Computing & Low-Level AI Systems

> **Systematic exploration of micro-architectural optimizations, analytical roofline modeling, and shared-memory parallel runtimes on modern multicore architectures.**

---

### Highlights & Empirical Breakthroughs

* **Compute Saturation:** Đạt **510.8 GFLOPS** (~33% Theoretical Peak) trên FP32 GEMM ($N=2016$) với **89.9% parallel scaling efficiency** trên 8 physical cores.
* **Cache Ping-Pong Elimination:** Triệt tiêu hoàn toàn False Sharing bằng `alignas(64)`, tăng tốc **101.57x** so với unaligned layout.
* **Workload Auto-Balancing:** Khắc phục barrier idle stalls do lệch tải bậc hai bằng `dynamic(64)`, nhanh hơn **11.3x** so với `static`.
* **Hardware Limit Profiling:** Đo đạc hoàn chỉnh băng thông đa tầng L1D/L2/L3/DRAM để dựng mô hình Cache-Aware Roofline Model (CARM).

---

## Architecture Blueprint

```text
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
```

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

### 1. General Matrix Multiplication (GEMM FP32, N = 2016)
Workload complexity: 2 * N^3 ≈ 16.38 GFLOPs.

| Implementation Stage | Core Optimization Primitives | Latency | Throughput | Speedup | Regime |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `01_naive_ijk` | Stride-N memory traversal, scalar | 1320.9 ms | 12.4 GFLOPS | 1.0x | DRAM Bound |
| `02_tiled_gemm` | Hierarchical blocking for L2/L3 residency | 148.9 ms | 110.0 GFLOPS | 8.9x | L3 Latency Bound |
| `03_avx2_naive` | Vectorized loop, unaligned strided loads | 66.8 ms | 245.0 GFLOPS | 19.8x | L2 Bandwidth Bound |
| **`04_blis_6x16`** | **2-Level Packing + 6x16 Register Tiling + OpenMP** | **32.08 ms** | **510.8 GFLOPS** | **41.2x** | **Compute Plateau** |

> **Analysis:** Việc chia nhỏ tile 6x16 giữ 12 thanh ghi vector %ymm tính toán liên tục, kết hợp pack mảng liên tục trên bộ nhớ đã loại bỏ hoàn toàn cache evictions, đẩy throughput lên vùng Compute-Bound của Roofline chart.

---

### 2. OpenMP Architectural Suite (`03-multithread-openmp`)

#### False Sharing Elimination (`01_false_sharing.cpp`)
* **Problem:** 8 luồng ghi đè đồng thời vào các biến cạnh nhau trong cùng 1 cache line 64-byte, gây nghẽn bus MESI liên tục.
* **Fix:** Cấp phát bộ nhớ đệm `alignas(64)` cho từng biến đếm của luồng.
* **Latency:** Giảm từ `0.4986 ms` xuống `0.0049 ms` (**Speedup: 101.57x**).

#### First-Touch Policy Profiling (`02_numa_first_touch.cpp`)
* **Observation:** Thử nghiệm cấp phát Serial vs Parallel trên mảng 512 MB chỉ chênh lệch `-0.39%` (không đáng kể).
* **Architectural Insight:** Chứng minh chip laptop là monolithic die chuẩn UMA (1 NUMA node), nơi mọi core truy cập RAM với khoảng cách và độ trễ đồng nhất.

#### Workload Balancing (`03_scheduling_overhead.cpp`)
* **Observation:** Với tải lệch hàm mũ (O(i^2) trên 20,000 tasks):
  * `static`: `0.2223 ms` (Cores nhận việc nhẹ phải chờ core cuối ở barrier).
  * `dynamic, 1`: `0.4825 ms` (Nghẽn atomic queue vì tranh chấp lấy việc liên tục).
  * `dynamic, 64`: **`0.0196 ms`** (**Nhanh hơn 11.3x so với static** nhờ triệt tiêu lock overhead).

---

## Quickstart & Reproducibility

```bash
# Clone and enter repo
git clone [https://github.com/](https://github.com/)<your-username>/HPC-Journey.git
cd HPC-Journey

# Run optimized BLIS GEMM benchmark
g++ -O3 -mavx2 -mfma -fopenmp -march=native \
    01-matrix-gemm-cpu/gemm_refactor.cpp -o gemm_bench
OMP_PLACES=cores OMP_PROC_BIND=close ./gemm_bench

# Run OpenMP micro-architectural suite
cd 03-multithread-openmp && make all
./01_false_sharing
./02_numa_first_touch
./03_scheduling_overhead

```
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