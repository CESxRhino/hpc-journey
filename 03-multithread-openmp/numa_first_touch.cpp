#include<iostream>
#include<vector>
#include<omp.h>
#include<chrono>
using namespace std;
constexpr size_t N = 128 * 1024 * 1024;
constexpr int REPEAT = 10;
int main(){
    omp_set_num_threads(8);
    cout << "=== NUMA FIRST-TOUCH BENCHMARK (Array: 512 MB) ===\n";
    //khoi tao don luong
    float *arr_bad = new float[N];
    for(size_t i = 0;i < N;i++){
        arr_bad[i] = 1.0f;
    }
    auto t0 = chrono::high_resolution_clock::now();
    for(int r = 0;r < REPEAT;r++){
        for(int i = 0;i < N;i++){
            arr_bad[i] = arr_bad[i] * 2.0f + 0.5f;
        }
    }
    auto t1 = chrono::high_resolution_clock::now();
    double time_bad = chrono::duration<double>(t1 - t0).count();
    cout << "Serial First-Touch compute time:   " << time_bad * 1000.0 << " ms\n";
    delete[] arr_bad;
    float *arr_good = new float[N];
    #pragma omp parallel for schedule(static)
    for(size_t i = 0;i < N;i++){
        arr_bad[i] = 1.0f;
    }
    auto t2 = chrono::high_resolution_clock::now();
    for(int r = 0;r < REPEAT;r++){
        for(int i = 0;i < N;i++){
            arr_good[i] = arr_good[i] * 2.0f + 0.5f;
        }
    }
    auto t3 = chrono::high_resolution_clock::now();
    double time_good = chrono::duration<double>(t3 - t2).count();
    cout << "Parallel First-Touch compute time:   " << time_good * 1000.0 << " ms\n";
    cout << "Latency Reduction:                 " << (time_bad - time_good) / time_bad * 100.0 << "%\n\n";
}