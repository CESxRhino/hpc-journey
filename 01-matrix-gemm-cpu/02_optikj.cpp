#include <bits/stdc++.h>
using namespace std;
int mod = 1e9 + 7;
const int N = 1024;
vector<float> a(N * N,1.0f);
vector<float> b(N * N,2.0f);
vector<float> c(N * N,0.0f);
void mat_mulikj()
{
    for(int i = 0;i < N;i++){
        for(int k = 0;k < N;k++){
            for(int j = 0;j < N;j++){
                c[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}
signed main()
{
    auto start = chrono::high_resolution_clock::now();
    mat_mulikj();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double,milli>duration = end - start;
    cout << "Time: " << duration.count() << "ms\n";
}