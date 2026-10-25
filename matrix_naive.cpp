#include <bits/stdc++.h>
using namespace std;
int mod = 1e9 + 7;
const int N = 1024;
vector<float> a(N * N,1.0f);
vector<float> b(N * N,2.0f);
vector<float> c(N * N,0.0f);
void mat_mul1()
{
    for(int i = 0;i < N;i++){
        for(int j = 0;j < N;j++){
            for(int k = 0;k < N;k++){
                c[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}
void mat_mul2()
{
    for(int i = 0;i < N;i++){
        for(int k = 0;k < N;k++){
            for(int j = 0;j < N;j++){
                c[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}
signed main(int argc,char *argv[])
{
    if(argc < 2) return 1;
    int mode = stoi(argv[1]);
    auto start = chrono::high_resolution_clock::now();
    if(mode == 1) mat_mul1();
    else mat_mul2();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double,milli>duration = end - start;
    cout << "Time: " << duration.count() << "ms\n";
}