#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>
#include <omp.h>
using namespace std;
const int N = 1024;
const int BLOCK_SIZE = 32;
vector<float> A(N * N, 1.0f);
vector<float> B(N * N, 2.0f);
vector<float> C(N * N, 0.0f);
const int MICRO_M = 4;
const int MICRO_N = 8;
void pack_A(const float* __restrict A, float* __restrict A_packed, int mc, int kc, int cur_N) {
    for (int i = 0; i < cur_N; i++) {
        for (int k = 0; k < cur_N; k++) {
            A_packed[i * BLOCK_SIZE + k] = A[(mc + i) * N + (kc + k)];
        }
    }
}
void pack_B(const float* __restrict B, float* __restrict B_packed, int kc, int nc, int cur_N) {
    for (int k = 0; k < cur_N; k++) {
        for (int j = 0; j < cur_N; j++) {
            B_packed[k * BLOCK_SIZE + j] = B[(kc + k) * N + (nc + j)];
        }
    }
}
inline void macro_kernel_avx(const float* A_block, const float* B_block, float* C_ptr, int cur_N) {
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    for (int k = 0; k < cur_N; ++k) {
        __m256 b_reg = _mm256_load_ps(&B_block[k * BLOCK_SIZE]);
        __m256 a0 = _mm256_set1_ps(A_block[0 * BLOCK_SIZE + k]);
        c0 = _mm256_fmadd_ps(a0, b_reg, c0);
        __m256 a1 = _mm256_set1_ps(A_block[1 * BLOCK_SIZE + k]);
        c1 = _mm256_fmadd_ps(a1, b_reg, c1);
        __m256 a2 = _mm256_set1_ps(A_block[2 * BLOCK_SIZE + k]);
        c2 = _mm256_fmadd_ps(a2, b_reg, c2);
        __m256 a3 = _mm256_set1_ps(A_block[3 * BLOCK_SIZE + k]);
        c3 = _mm256_fmadd_ps(a3, b_reg, c3);
    }
    _mm256_storeu_ps(&C_ptr[0 * N], _mm256_add_ps(_mm256_loadu_ps(&C_ptr[0 * N]), c0));
    _mm256_storeu_ps(&C_ptr[1 * N], _mm256_add_ps(_mm256_loadu_ps(&C_ptr[1 * N]), c1));
    _mm256_storeu_ps(&C_ptr[2 * N], _mm256_add_ps(_mm256_loadu_ps(&C_ptr[2 * N]), c2));
    _mm256_storeu_ps(&C_ptr[3 * N], _mm256_add_ps(_mm256_loadu_ps(&C_ptr[3 * N]), c3));
}
void gemm_optimized(const float* A, const float* B, float* C) {
    #pragma omp parallel
    {
        float* A_packed = (float*)aligned_alloc(32, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
        float* B_packed = (float*)aligned_alloc(32, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
        #pragma omp for schedule(dynamic)
        for (int mc = 0; mc < N; mc += BLOCK_SIZE) {
            int cur_M = min(BLOCK_SIZE, N - mc);
            for (int kc = 0; kc < N; kc += BLOCK_SIZE) {
                int cur_K = min(BLOCK_SIZE, N - kc);
                pack_A(A, A_packed, mc, kc, cur_M);
                for (int nc = 0; nc < N; nc += BLOCK_SIZE) {
                    int cur_N = min(BLOCK_SIZE, N - nc);
                    pack_B(B, B_packed, kc, nc,cur_N);
                    for (int mr = 0; mr < cur_M; mr += MICRO_M) {
                        for (int nr = 0; nr < cur_N; nr += MICRO_N) {
                            macro_kernel_avx(
                                &A_packed[mr * BLOCK_SIZE], 
                                &B_packed[nr], 
                                &C[(mc + mr) * N + (nc + nr)], 
                                cur_K
                            );
                        }
                    }
                }
            }
        }
        free(A_packed);
        free(B_packed);
    }
}
bool verify(const float* C, float expected, int size) {
    for(int i = 0; i < size; ++i) {
        if(abs(C[i] - expected) > 1e-2f) {
            cout << "ERROR At " << i << ": Value = " << C[i] << ", expected = " << expected << endl;
            return false;
        }
    }
    return true;
}
int main() {
    auto start = chrono::high_resolution_clock::now();
    gemm_optimized(A.data(), B.data(), C.data());
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> duration = end - start;
    cout << "Thoi gian xu ly: " << duration.count() << " ms" << std::endl;
    float expected = (float)N * 1.0f * 2.0f;
    if(verify(C.data(), expected, N * N)) {
        cout << "Result Verification: SUCCESS !" << endl;
    }
    return 0;
}