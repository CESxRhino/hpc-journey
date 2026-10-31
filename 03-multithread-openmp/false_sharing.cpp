#include<iostream>
#include<vector>
#include<omp.h>
#include<chrono>
using namespace std;
#define int long long
constexpr int ITERS = 500'000'000LL;
constexpr int MAX_THREADS = 8;
struct BadCounter{
    int val{0};
};
// vector<BadCounter> bd;
struct alignas(64) GoodCount{
    int val{0};
};
signed main(){
    int num_thr = 8;
    omp_set_num_threads(num_thr);
    vector<BadCounter> bd(num_thr);
    vector<GoodCount> gd(num_thr);
    cout << "=== FALSE SHARING BENCHMARK (" << num_thr << " threads) ===\n";
    auto t0 = chrono::high_resolution_clock::now();
    //False Sharing
    #pragma omp parallel
    {
        int thrid = omp_get_thread_num();
        for(int i = 0;i < ITERS;++i){
            bd[thrid].val++;
        }
    }
    auto t1 = chrono::high_resolution_clock::now();
    double time_bad = chrono::duration<double>(t1 - t0).count();
    cout << "With False Sharing:    " << time_bad * 1000.0 << " ms\n";  
    //Line Padding
    auto t2 = chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        int thrid = omp_get_thread_num();
        for(int i = 0;i < ITERS;++i){
            gd[thrid].val++;
        }
    }
    auto t3 = chrono::high_resolution_clock::now();
    double time_good = chrono::duration<double>(t3 - t2).count();
    cout << "With 64-byte Padding:    " << time_good * 1000.0 << " ms\n";
    cout << "Speedup:               " << time_bad / time_good << "x\n\n";  
}