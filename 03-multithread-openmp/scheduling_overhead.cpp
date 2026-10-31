#include<iostream>
#include<vector>
#include<chrono>
#include<omp.h>
#include<cmath>
using namespace std;
constexpr int TASKS = 20000;
inline double heavy_work(int i){
    double res = 0.0;
    int iters = (int)pow((double)i/TASKS,2) * 50000.0; 
    for(int k = 0;k < iters;k++){
        res += sin(k) * 0.5f;
    }
    return res;
}
int main(){
    omp_set_num_threads(8);
    cout << "=== SCHEDULING STRATEGY COMPARISON ===\n";
    auto t0 = chrono::high_resolution_clock::now();
    double sum1 = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:sum1)
    for (int i = 0; i < TASKS; ++i) {
        sum1 += heavy_work(i);
    }
    auto t1 = chrono::high_resolution_clock::now();
    cout << "schedule(static):             "<< chrono::duration<double>(t1 - t0).count() * 1000.0 << " ms\n";
    double time_bad = chrono::duration<double>(t1 - t0).count();
    auto t2 = chrono::high_resolution_clock::now();
    double sum2 = 0.0;
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:sum2)
    for (int i = 0; i < TASKS; ++i) {
        sum2 += heavy_work(i);
    }
    auto t3 = chrono::high_resolution_clock::now();   
    cout << "schedule(dynamic, 1):         " << chrono::duration<double>(t3 - t2).count() * 1000.0 << " ms\n";
    auto t4 = chrono::high_resolution_clock::now();
    double sum3 = 0.0;
    #pragma omp parallel for schedule(dynamic, 64) reduction(+:sum3)
    for (int i = 0; i < TASKS; ++i) {
        sum2 += heavy_work(i);
    }
    auto t5 = chrono::high_resolution_clock::now();   
    cout << "schedule(dynamic, 64):         " << chrono::duration<double>(t5 - t4).count() * 1000.0 << " ms\n";
    auto t6 = chrono::high_resolution_clock::now();
    double sum4 = 0.0;
    #pragma omp parallel for schedule(dynamic, 16) reduction(+:sum4)
    for (int i = 0; i < TASKS; ++i) {
        sum4 += heavy_work(i);
    }
    auto t7 = chrono::high_resolution_clock::now();   
    cout << "schedule(dynamic, 16):         " << chrono::duration<double>(t7 - t6).count() * 1000.0 << " ms\n";
}