#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <omp.h>
#include <new>
#include<cstdlib>
#include<fstream>
using namespace std;
double Peak_FPP(uint64_t iterations){
    int max_threads = omp_get_max_threads(); 
    double start_time = omp_get_wtime();
    #pragma omp parallel
    {
        __m256d r0 = _mm256_set1_pd(1.0);
        __m256d r1 = _mm256_set1_pd(1.0);
        __m256d r2 = _mm256_set1_pd(1.0);
        __m256d r3 = _mm256_set1_pd(1.0);
        __m256d mul = _mm256_set1_pd(1.0000001);
        for (uint64_t i = 0; i < iterations; ++i) {
            r0 = _mm256_fmadd_pd(r0, mul, mul);
            r1 = _mm256_fmadd_pd(r1, mul, mul);
            r2 = _mm256_fmadd_pd(r2, mul, mul);
            r3 = _mm256_fmadd_pd(r3, mul, mul);
        }
        volatile double sink = ((double*)&r0)[0] + ((double*)&r1)[0];
        (void)sink;
    }
    double duration = omp_get_wtime() - start_time;
    double total_flops = static_cast<double>(iterations) * 32.0 * max_threads;
    return (total_flops / duration) / 1e9;
}
double Peak_Mem_Bandwith(int tot_ele, int benchmark_reps){
    double* a = static_cast<double*>(aligned_alloc(64, tot_ele * sizeof(double)));
    double* b = static_cast<double*>(aligned_alloc(64, tot_ele * sizeof(double)));
    double* c = static_cast<double*>(aligned_alloc(64, tot_ele * sizeof(double)));
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < tot_ele; ++i) {
        b[i] = 1.0;
        c[i] = 2.0;
        a[i] = 0.0;
    }
    double scalar = 3.0;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < tot_ele; ++i){
        a[i] = b[i] + scalar * c[i];
    }
    double start_time = omp_get_wtime();
    for(int r = 0; r < benchmark_reps; ++r) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < tot_ele; ++i) {
            a[i] = b[i] + scalar * c[i];
        }
    }
    double duration = omp_get_wtime() - start_time;
    double total_bytes = static_cast<double>(tot_ele) * 3.0 * sizeof(double) * benchmark_reps;
    double bw_gbs = (total_bytes / duration) / 1e9;
    free(a);
    free(b);
    free(c);
    return bw_gbs;
}
double measure_l1_bandwidth(int reps) {
    int num_threads = omp_get_max_threads();
    constexpr size_t PER_THREAD_N = 2048;
    double scalar = 3.0;
    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        alignas(64) double a[PER_THREAD_N];
        alignas(64) double b[PER_THREAD_N];
        alignas(64) double c[PER_THREAD_N];
        for (size_t i = 0; i < PER_THREAD_N; ++i) {
            b[i] = 1.0; c[i] = 2.0; a[i] = 0.0;
        }
        for (int r = 0; r < reps; ++r) {
            #pragma omp simd
            for (size_t i = 0; i < PER_THREAD_N; ++i) {
                a[i] = b[i] + scalar * c[i];
            }
        }
        volatile double sink = a[0];
        (void)sink;
    }
    double duration = omp_get_wtime() - t0;
    double total_bytes = static_cast<double>(PER_THREAD_N * num_threads) * 3.0 * sizeof(double) * reps;
    return (total_bytes / duration) / 1e9;
}
int main(){
    int threads = omp_get_max_threads();
    cout << "Threads used: " << threads << endl;
    cout << "Benchmarking hardware limits...\n";
    double p_max = Peak_FPP(2'000'000'000ULL);
    cout << "-> Peak Compute (P_max): " << p_max << " GFLOPS\n";
    cout << "Measuring L1 Bandwidth...\n";
    double b_l1 = measure_l1_bandwidth(5'000'000);
    cout << "-> Peak DRAM Bandwidth (b_max): " << b_l1 << " GB/s\n";
    cout << "Measuring L2 Bandwidth...\n";
    size_t l2_elements = 100'000;
    double b_l2 = Peak_Mem_Bandwith(l2_elements, 200'000);
    cout << "L2 Bandwidth: " << b_l2 << " GB/s\n";
    cout << "Measuring L3 Bandwidth...\n";
    size_t l3_elements = 400'000;
    double b_l3 = Peak_Mem_Bandwith(l3_elements, 20'000);
    cout << "L3 Bandwidth: " << b_l3 << " GB/s\n";
    cout << "Measuring DRAM Bandwidth...\n";
    size_t dram_elements = 64'000'000;
    double b_dram = Peak_Mem_Bandwith(dram_elements, 5);
    cout << "DRAM Bandwidth: " << b_dram << " GB/s\n";
    ofstream out("carm_data.json");
    out << "{\n";
    out << "  \"p_max\": " << p_max << ",\n";
    out << "  \"b_l1\": " << b_l1 << ",\n";
    out << "  \"b_l2\": " << b_l2 << ",\n";
    out << "  \"b_l3\": " << b_l3 << ",\n";
    out << "  \"b_dram\": " << b_dram << "\n";
    out << "}\n";
    out.close();
    cout << "Results saved to carm_data.json\n";
    return 0;
}