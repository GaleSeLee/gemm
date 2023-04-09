#include "gemm.h"
#include "cblas.h"

extern int M, N, K;
void gemm_openblas(float *A, float *B, float *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                M, N, K, 1, A, K, B, N, 0, C, N);
}