#include <arm_neon.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>

using namespace std;

// Macros for matrix access
#define IDX_A(i, j) a[(i) * lda + (j)]
#define IDX_B(i, j) b[(i) * ldb + (j)]
#define IDX_C(i, j) c[(i) * ldc + (j)]

// Raw implementation: Naive triple loop
void sgemm_raw(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += IDX_A(i, p) * IDX_B(p, j);
            }
            IDX_C(i, j) += sum;
        }
    }
}

// NEON implementation: 4x4 kernel from MMult_4x4_11.cpp
void AddDot4x4(int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {
  float *a_0p_pntr, *a_1p_pntr, *a_2p_pntr, *a_3p_pntr;

  a_0p_pntr = &IDX_A(0, 0);
  a_1p_pntr = &IDX_A(1, 0);
  a_2p_pntr = &IDX_A(2, 0);
  a_3p_pntr = &IDX_A(3, 0);

  float32x4_t c_0p_sum = { 0 };
  float32x4_t c_1p_sum = { 0 };
  float32x4_t c_2p_sum = { 0 };
  float32x4_t c_3p_sum = { 0 };

  float a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg;

  for (int p = 0; p < k; ++p) {
    float32x4_t b_reg = vld1q_f32(b);
    b += 4;

    a_0p_reg = *a_0p_pntr++;
    a_1p_reg = *a_1p_pntr++;
    a_2p_reg = *a_2p_pntr++;
    a_3p_reg = *a_3p_pntr++;

    c_0p_sum = vmlaq_n_f32(c_0p_sum, b_reg, a_0p_reg);
    c_1p_sum = vmlaq_n_f32(c_1p_sum, b_reg, a_1p_reg);
    c_2p_sum = vmlaq_n_f32(c_2p_sum, b_reg, a_2p_reg);
    c_3p_sum = vmlaq_n_f32(c_3p_sum, b_reg, a_3p_reg);
  }

  float *c_pntr = 0;
  c_pntr = &IDX_C(0, 0);
  float32x4_t c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_0p_sum);
  vst1q_f32(c_pntr, c_reg);

  c_pntr = &IDX_C(1, 0);
  c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_1p_sum);
  vst1q_f32(c_pntr, c_reg);

  c_pntr = &IDX_C(2, 0);
  c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_2p_sum);
  vst1q_f32(c_pntr, c_reg);

  c_pntr = &IDX_C(3, 0);
  c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_3p_sum);
  vst1q_f32(c_pntr, c_reg);
}

void PackMatrixB(int k, float *b, int ldb, float *b_to) {
  int j;
  for (j = 0; j < k; ++j) {
    float *b_ij_pntr = &IDX_B(j, 0);
    *b_to++ = b_ij_pntr[0];
    *b_to++ = b_ij_pntr[1];
    *b_to++ = b_ij_pntr[2];
    *b_to++ = b_ij_pntr[3];
  }
}

void InnerKernel(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {
  int i, j;
  float packedB[k * n];

  for (j = 0; j < n; j += 4) {
    PackMatrixB(k, &IDX_B(0, j), ldb, &packedB[j * k]);
    for (i = 0; i < m; i += 4) {
      AddDot4x4(k, &IDX_A(i, 0), lda, &packedB[j * k], 4, &IDX_C(i, j), ldc);
    }
  }
}

void sgemm_neon(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {
    #define mc 256
    #define kc 128
    #define min(i, j) ((i) < (j) ? (i) : (j))

    int i, p, pb, ib;
    for (p = 0; p < k; p += kc) {
        pb = min(k - p, kc);
        for (i = 0; i < m; i += mc) {
            ib = min(m - i, mc);
            InnerKernel(ib, n, pb, &IDX_A(i, p), lda, &IDX_B(p, 0), ldb, &IDX_C(i, 0), ldc);
        }
    }
}

int main(){
    const int M = 512;
    const int N = 512;
    const int K = 512;
    
    vector<float> A(M * K);
    vector<float> B(K * N);
    vector<float> C_raw(M * N, 0.0f);
    vector<float> C_neon(M * N, 0.0f);

    // Random generation
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f, 1.0f);

    generate(A.begin(), A.end(), [&]() { return dis(gen); });
    generate(B.begin(), B.end(), [&]() { return dis(gen); });

    // Warmup
    sgemm_raw(M, N, K, A.data(), K, B.data(), N, C_raw.data(), N);
    sgemm_neon(M, N, K, A.data(), K, B.data(), N, C_neon.data(), N);
    
    // Reset C
    fill(C_raw.begin(), C_raw.end(), 0.0f);
    fill(C_neon.begin(), C_neon.end(), 0.0f);

    // Benchmark Raw
    auto start_raw = chrono::high_resolution_clock::now();
    sgemm_raw(M, N, K, A.data(), K, B.data(), N, C_raw.data(), N);
    auto end_raw = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration_raw = end_raw - start_raw;

    // Benchmark Neon
    auto start_neon = chrono::high_resolution_clock::now();
    sgemm_neon(M, N, K, A.data(), K, B.data(), N, C_neon.data(), N);
    auto end_neon = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration_neon = end_neon - start_neon;

    cout << "Matrix size: " << M << "x" << N << "x" << K << endl;
    cout << "Raw time: " << duration_raw.count() << " ms" << endl;
    cout << "Neon time: " << duration_neon.count() << " ms" << endl;

    // Verification
    bool correct = true;
    for(int i=0; i<M*N; ++i) {
        if (abs(C_raw[i] - C_neon[i]) > 1e-3) {
            correct = false;
            cout << "Mismatch at " << i << ": " << C_raw[i] << " vs " << C_neon[i] << endl;
            break;
        }
    }
    if(correct) cout << "Results match!" << endl;

    return 0;
}
