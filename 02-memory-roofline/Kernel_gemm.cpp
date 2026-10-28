#include<iostream>
#include<omp.h>
#include<vector>
#include<algorithm>
using namespace std;
const int N = 1024;
const int BLOCK_SIZE = 64;
void mat_mulijk(const float* a,const float* b,float *c)
{
    for(int i = 0;i < N;i++){
        for(int j = 0;j < N;j++){
            for(int k = 0;k < N;k++){
                c[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}
void cache_tilt(const float* a2,const float* b2,float *c2){
    #pragma omp parallel for collapse(2)
    for(int sj = 0;sj < N; sj += BLOCK_SIZE){
        for(int si = 0;si < N;si += BLOCK_SIZE){
            for(int sk = 0;sk < N;sk += BLOCK_SIZE){
                for(int i = si;i < min(N,si + BLOCK_SIZE);i++){
                    for(int k = sk;k < min(N,sk + BLOCK_SIZE);k++){
                        float regisA = a2[i * N + k];
                        for(int j = sj;j < min(N,sj + BLOCK_SIZE);j++){
                            c2[i * N + j] += regisA * b2[k * N + j];
                        }
                    }
                }
            }
        }
    }
}
int main(){
    vector<float> a(N * N,2.0f); 
    vector<float> b(N * N,1.0f); 
    vector<float> c1(N * N,0.0f); 
    vector<float> c2(N * N,0.0f); 
    double tot_flops = 2.0 * N * N * N;
    double t0 = omp_get_wtime();
    mat_mulijk(a.data(),b.data(),c1.data());
    double t_naive = omp_get_wtime() - t0;
    double gflops_naive = (tot_flops / t_naive) / 1e9;
    double t1 = omp_get_wtime();
    cache_tilt(a.data(), b.data(), c2.data());
    double t_blocked = omp_get_wtime() - t1;
    double gflops_blocked = (tot_flops / t_blocked) / 1e9;
    cout << "[GEMM N=" << N << "]\n";
    cout << "Naive GEMM:   " << gflops_naive << " GFLOPS (Time: " << t_naive << "s)\n";
    cout << "Blocked GEMM: " << gflops_blocked << " GFLOPS (Time: " << t_blocked << "s)\n";
}