#include <bits/stdc++.h>
using namespace std;
int mod = 1e9 + 7;
const int N = 1024;
vector<float> a1(N * N,1.0f);
vector<float> b1(N * N,2.0f);
vector<float> c1(N * N,0.0f);
vector<float> a2(N * N,1.0f);
vector<float> b2(N * N,2.0f);
vector<float> c2(N * N,0.0f);
const int BLOCK_SIZE = 32;
void mat_mulijk()
{
    for(int i = 0;i < N;i++){
        for(int j = 0;j < N;j++){
            for(int k = 0;k < N;k++){
                c1[i * N + j] += a1[i * N + k] * b1[k * N + j];
            }
        }
    }
}
void cache_tilt(){
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
bool ver_res(){
    for(int i = 0;i < N * N;i++){
        if(abs(c1[i] - c2[i]) > 1e-2f){
            cout << "False at index " << i << ": " << c1[i] << " vs " << c2[i] << endl;
            return false;
        }
    }
    return true;
}
signed main()
{
    auto start = chrono::high_resolution_clock::now();
    mat_mulijk();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double,milli>duration1 = end - start;
    cout << "[2] Naive GEMM Time: " << duration1.count() << " ms" << endl;
    start = chrono::high_resolution_clock::now();
    cache_tilt();
    end = chrono::high_resolution_clock::now();
    chrono::duration<double,milli>duration2 = end - start;
    cout << "[3] Tilted_Cache GEMM Time: " << duration2.count() << " ms" << endl;
    if(ver_res()){
        double speedup = duration1.count()/duration2.count();
        cout << "\nResult Verification: SUCCESS!" << endl;
        cout << "==> SPEEDUP: " << speedup << " faster" << endl;
    }
    return 0;
}
