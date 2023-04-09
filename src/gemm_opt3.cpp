#include "gemm.h"

#define KKDIM 16
#define ptr(ii, jj, ldb, mat) mat+(ii)*ldb+(jj)

extern int M, N, K;
void gemm_opt3(float *A, float *B, float *C) {
    float *b_ptr_00, *b_ptr_01, *b_ptr_02, *b_ptr_03;
    float *a_ptr_00, *a_ptr_10, *a_ptr_20, *a_ptr_30;
    float *c_ptr_00, *c_ptr_01, *c_ptr_02, *c_ptr_03,
        *c_ptr_10, *c_ptr_11, *c_ptr_12, *c_ptr_13,
        *c_ptr_20, *c_ptr_21, *c_ptr_22, *c_ptr_23,
        *c_ptr_30, *c_ptr_31, *c_ptr_32, *c_ptr_33;
    int up_sub_kk;

    for (int sub_kk = 0; sub_kk < K; sub_kk += KKDIM) {
        up_sub_kk = sub_kk + KKDIM;
        for (int ii = 0; ii < M; ii+=4) {
            for (int kk = sub_kk; kk < up_sub_kk; kk++) {
                for (int jj = 0; jj < N; jj+=4) {
                    c_ptr_00 = ptr(ii, jj, N, C);
                    c_ptr_01 = c_ptr_00 + 1;
                    c_ptr_02 = c_ptr_01 + 1;
                    c_ptr_03 = c_ptr_02 + 1;
                    c_ptr_10 = c_ptr_00 + N;
                    c_ptr_11 = c_ptr_10 + 1;
                    c_ptr_12 = c_ptr_10 + 2;
                    c_ptr_13 = c_ptr_10 + 3;
                    c_ptr_20 = c_ptr_10 + N;
                    c_ptr_21 = c_ptr_20 + 1;
                    c_ptr_22 = c_ptr_20 + 2;
                    c_ptr_23 = c_ptr_20 + 3;
                    c_ptr_30 = c_ptr_20 + N;
                    c_ptr_31 = c_ptr_30 + 1;
                    c_ptr_32 = c_ptr_30 + 2;
                    c_ptr_33 = c_ptr_30 + 3;
                    b_ptr_00 = ptr(kk, jj, N, B);
                    b_ptr_01 = b_ptr_00 + 1;
                    b_ptr_02 = b_ptr_01 + 1;
                    b_ptr_03 = b_ptr_02 + 1;
                    a_ptr_00 = ptr(ii, kk, K, A);
                    a_ptr_10 = a_ptr_00 + K;
                    a_ptr_20 = a_ptr_10 + K;
                    a_ptr_30 = a_ptr_20 + K;
                    *c_ptr_00 += *a_ptr_00 * *b_ptr_00;
                    *c_ptr_10 += *a_ptr_10 * *b_ptr_00;
                    *c_ptr_20 += *a_ptr_20 * *b_ptr_00;
                    *c_ptr_30 += *a_ptr_30 * *b_ptr_00;
                    *c_ptr_01 += *a_ptr_00 * *b_ptr_01;
                    *c_ptr_11 += *a_ptr_10 * *b_ptr_01;
                    *c_ptr_21 += *a_ptr_20 * *b_ptr_01;
                    *c_ptr_31 += *a_ptr_30 * *b_ptr_01;
                    *c_ptr_02 += *a_ptr_00 * *b_ptr_02;
                    *c_ptr_12 += *a_ptr_10 * *b_ptr_02;
                    *c_ptr_22 += *a_ptr_20 * *b_ptr_02;
                    *c_ptr_32 += *a_ptr_30 * *b_ptr_02;
                    *c_ptr_03 += *a_ptr_00 * *b_ptr_03;
                    *c_ptr_13 += *a_ptr_10 * *b_ptr_03;
                    *c_ptr_23 += *a_ptr_20 * *b_ptr_03;
                    *c_ptr_33 += *a_ptr_30 * *b_ptr_03;
                }
            }
        }
    }
}