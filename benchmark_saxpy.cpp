#include <arm_neon.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>

using namespace std;

void saxpy_neon(float alpha, float * X, float* Y, const unsigned int N){
    // DO SOMETHING
}

void saxpy_raw(float alpha, float * X, float* Y, const unsigned int N){
    for (unsigned int i = 0; i < N; ++i) {
        Y[i] += alpha * X[i];
    }
}


int main(){
    const unsigned int N = 1024 * 1024 * 16;
    const float alpha = 2.5f;
    
    vector<float> X(N);
    vector<float> Y_raw(N);
    vector<float> Y_neon(N);

    // Random generation
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f, 1.0f);

    generate(X.begin(), X.end(), [&]() { return dis(gen); });
    generate(Y_raw.begin(), Y_raw.end(), [&]() { return dis(gen); });
    Y_neon = Y_raw; // Copy initial Y

    // Warmup
    saxpy_raw(alpha, X.data(), Y_raw.data(), N);
    saxpy_neon(alpha, X.data(), Y_neon.data(), N);
    
    // Reset Y for benchmark (though not strictly necessary for performance, good for correctness check if we were to re-run)
    // For benchmarking, we just run it again. The values will change but performance should be same.
    // Let's re-init Y to keep values bounded if we ran multiple times, but here once is fine.
    
    // Benchmark Raw
    auto start_raw = chrono::high_resolution_clock::now();
    saxpy_raw(alpha, X.data(), Y_raw.data(), N);
    auto end_raw = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration_raw = end_raw - start_raw;

    // Benchmark Neon
    auto start_neon = chrono::high_resolution_clock::now();
    saxpy_neon(alpha, X.data(), Y_neon.data(), N);
    auto end_neon = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration_neon = end_neon - start_neon;

    cout << "Data size: " << N << " elements" << endl;
    cout << "Raw time: " << duration_raw.count() << " ms" << endl;
    cout << "Neon time: " << duration_neon.count() << " ms" << endl;

    // Verification
    // Note: Since we ran the function twice (warmup + bench), Y has been updated twice.
    // We need to ensure Y_raw and Y_neon started from same state before EACH run if we want to compare exact results 
    // OR just compare them now assuming they did same operations.
    // They both did: Y = Y_init + alpha*X (warmup) -> Y = Y_prev + alpha*X (bench).
    // So they should match.
    
    bool correct = true;
    for(unsigned int i=0; i<N; ++i) {
        if (abs(Y_raw[i] - Y_neon[i]) > 1e-4) { // Slightly larger tolerance for FMA differences
            correct = false;
            cout << "Mismatch at " << i << ": " << Y_raw[i] << " vs " << Y_neon[i] << endl;
            break;
        }
    }
    if(correct) cout << "Results match!" << endl;

    return 0;
}
