#include "gemm.h"

#define idx(ii, jj, lda, mat) mat[(ii)*lda+(jj)]

extern int M, N, K;
void gemm_opt1(float *A, float *B, float *C) {
    for (int ii = 0; ii < M; ii+=4) {
        for (int jj = 0; jj < N; jj+=4) {
            for (int kk = 0; kk < K; kk++) {
                idx(ii, jj, N, C) += idx(ii, kk, K, A) * idx(kk, jj, N, B);
                idx(ii+1, jj, N, C) += idx(ii+1, kk, K, A) * idx(kk, jj, N, B);
                idx(ii+2, jj, N, C) += idx(ii+2, kk, K, A) * idx(kk, jj, N, B);
                idx(ii+3, jj, N, C) += idx(ii+3, kk, K, A) * idx(kk, jj, N, B);
                idx(ii, jj+1, N, C) += idx(ii, kk, K, A) * idx(kk, jj+1, N, B);
                idx(ii+1, jj+1, N, C) += idx(ii+1, kk, K, A) * idx(kk, jj+1, N, B);
                idx(ii+2, jj+1, N, C) += idx(ii+2, kk, K, A) * idx(kk, jj+1, N, B);
                idx(ii+3, jj+1, N, C) += idx(ii+3, kk, K, A) * idx(kk, jj+1, N, B);
                idx(ii, jj+2, N, C) += idx(ii, kk, K, A) * idx(kk, jj+2, N, B);
                idx(ii+1, jj+2, N, C) += idx(ii+1, kk, K, A) * idx(kk, jj+2, N, B);
                idx(ii+2, jj+2, N, C) += idx(ii+2, kk, K, A) * idx(kk, jj+2, N, B);
                idx(ii+3, jj+2, N, C) += idx(ii+3, kk, K, A) * idx(kk, jj+2, N, B);
                idx(ii, jj+3, N, C) += idx(ii, kk, K, A) * idx(kk, jj+3, N, B);
                idx(ii+1, jj+3, N, C) += idx(ii+1, kk, K, A) * idx(kk, jj+3, N, B);
                idx(ii+2, jj+3, N, C) += idx(ii+2, kk, K, A) * idx(kk, jj+3, N, B);
                idx(ii+3, jj+3, N, C) += idx(ii+3, kk, K, A) * idx(kk, jj+3, N, B);
            }
        }
    }
}