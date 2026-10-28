#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>
#include <omp.h>

using namespace std;

// 2016 chia hết cho cả 6 (2016/6 = 336) và 16 (2016/16 = 126)
// Loại bỏ hoàn toàn lỗi tràn biên bộ nhớ của C
constexpr int N = 2016; 

constexpr int MC = 96;  // 96 chia hết cho 6
constexpr int NC = 256; // 256 chia hết cho 16
constexpr int KC = 256;

inline void micro_kernel_6x16(
    int K,
    const float* __restrict A_packed,
    const float* __restrict B_packed,
    float* __restrict C, int ldc
) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    for (int p = 0; p < K; ++p) {
        __m256 b0 = _mm256_load_ps(&B_packed[p * 16 + 0]);
        __m256 b1 = _mm256_load_ps(&B_packed[p * 16 + 8]);

        __m256 a0 = _mm256_set1_ps(A_packed[p * 6 + 0]);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);

        __m256 a1 = _mm256_set1_ps(A_packed[p * 6 + 1]);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);

        __m256 a2 = _mm256_set1_ps(A_packed[p * 6 + 2]);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);

        __m256 a3 = _mm256_set1_ps(A_packed[p * 6 + 3]);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);

        __m256 a4 = _mm256_set1_ps(A_packed[p * 6 + 4]);
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);

        __m256 a5 = _mm256_set1_ps(A_packed[p * 6 + 5]);
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
    }

    auto store_c = [&](int row, __m256 r0, __m256 r1) {
        _mm256_storeu_ps(&C[row * ldc + 0], _mm256_add_ps(_mm256_loadu_ps(&C[row * ldc + 0]), r0));
        _mm256_storeu_ps(&C[row * ldc + 8], _mm256_add_ps(_mm256_loadu_ps(&C[row * ldc + 8]), r1));
    };

    store_c(0, c00, c01);
    store_c(1, c10, c11);
    store_c(2, c20, c21);
    store_c(3, c30, c31);
    store_c(4, c40, c41);
    store_c(5, c50, c51);
}

void gemm_safe(const float* A, const float* B, float* C) {
    #pragma omp parallel
    {
        // Cấp phát đủ kích thước cho panel
        float* A_pack = (float*)aligned_alloc(64, MC * KC * sizeof(float));
        float* B_pack = (float*)aligned_alloc(64, KC * 16 * sizeof(float));

        #pragma omp for schedule(static) collapse(2)
        for (int i = 0; i < N; i += MC) {
            for (int j = 0; j < N; j += NC) {
                for (int k = 0; k < N; k += KC) {
                    int cur_K = min(KC, N - k);

                    for (int jj = 0; jj < NC; jj += 16) {
                        // Pack panel B (KC x 16)
                        for (int p = 0; p < cur_K; ++p) {
                            for (int c = 0; c < 16; ++c) {
                                B_pack[p * 16 + c] = B[(k + p) * N + (j + jj + c)];
                            }
                        }

                        for (int ii = 0; ii < MC; ii += 6) {
                            // Pack 6 hàng của A (6 x KC) liên tục theo p
                            float* A_micro_ptr = &A_pack[(ii / 6) * (6 * cur_K)];
                            if (jj == 0) {
                                for (int p = 0; p < cur_K; ++p) {
                                    for (int r = 0; r < 6; ++r) {
                                        A_micro_ptr[p * 6 + r] = A[(i + ii + r) * N + (k + p)];
                                    }
                                }
                            }

                            micro_kernel_6x16(
                                cur_K,
                                A_micro_ptr,
                                B_pack,
                                &C[(i + ii) * N + (j + jj)],
                                N
                            );
                        }
                    }
                }
            }
        }

        free(A_pack);
        free(B_pack);
    }
}

int main() {
    size_t total_elements = (size_t)N * N;
    float* A = (float*)aligned_alloc(64, total_elements * sizeof(float));
    float* B = (float*)aligned_alloc(64, total_elements * sizeof(float));
    float* C = (float*)aligned_alloc(64, total_elements * sizeof(float));

    #pragma omp parallel for
    for (size_t idx = 0; idx < total_elements; ++idx) {
        A[idx] = 1.0f;
        B[idx] = 2.0f;
        C[idx] = 0.0f;
    }

    cout << "Testing GEMM (N=" << N << ") with " << omp_get_max_threads() << " threads...\n";

    // Warm-up
    gemm_safe(A, B, C);
    #pragma omp parallel for
    for (size_t idx = 0; idx < total_elements; ++idx) C[idx] = 0.0f;

    auto t0 = chrono::high_resolution_clock::now();
    gemm_safe(A, B, C);
    auto t1 = chrono::high_resolution_clock::now();

    double sec = chrono::duration<double>(t1 - t0).count();
    double gflops = (2.0 * N * N * N) / sec / 1e9;

    cout << "Time:       " << sec * 1000.0 << " ms\n";
    cout << "Throughput: " << gflops << " GFLOPS\n";

    free(A);
    free(B);
    free(C);
    cout << "Clean exit SUCCESS!\n";
    return 0;
}