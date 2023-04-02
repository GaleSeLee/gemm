#include "gemm.h"

extern int M, K, N;
void gemm_baseline(int *A, int *B, int *C) {
    for (int ii = 0; ii < M; ii ++) {
        for(int jj = 0; jj < N; jj++) {
            C[ii*N+jj] = 0;
            for (int kk = 0; kk < K; kk++) {
                C[ii*N+jj] += A[ii*K+kk] * B[kk*N+jj];
            }
        }
    }
}