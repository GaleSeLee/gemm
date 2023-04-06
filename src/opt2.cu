// 2060
// 3891.3
// 60.3208%
#include "gemm.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define TPBX 16
#define TPBY 16

__global__
void opt2_kernel(float *A, float *B, float *C, int M, int K, int N) {
    const int tx = threadIdx.x;
    const int block_idx_a_y = tx & 15;
    const int block_idx_a_x = ((tx>>5)<<3) + ((tx&31)>>4);
    const int block_idx_a = block_idx_a_x * 16 + block_idx_a_y;
    const int global_idx_a_x = block_idx_a_x + blockIdx.x * 64;
    const int global_idx_a_y = block_idx_a_y + blockIdx.y * 16;
    const int global_idx_a = global_idx_a_x * K + global_idx_a_y;

    // store idx
    const int block_idx_b_y = tx & 63;
    const int block_idx_b_x = tx >> 6 << 2;
    const int block_idx_b = block_idx_b_x * 64 + block_idx_b_y;
    // load idx
    const int global_idx_b_x = block_idx_b_x + blockIdx.x * 16;
    const int global_idx_b_y = block_idx_b_y + blockIdx.y * 64;
    const int global_idx_b = global_idx_b_x * N + global_idx_b_y;

    const int block_idx_c_y = ((tx & 15) << 2);
    const int block_idx_c_x = ((tx >> 4) << 2);
    const int global_idx_c_x = block_idx_c_x + blockIdx.x * 64;
    const int global_idx_c_y = block_idx_c_y + blockIdx.y * 64;
    const int global_idx_c = global_idx_c_x * N + global_idx_c_y;
    __shared__ float block_A[1024];
    __shared__ float block_B[1024];
    
    float tmp_00 = 0.0f;
    float tmp_10 = 0.0f;
    float tmp_20 = 0.0f;
    float tmp_30 = 0.0f;
    float tmp_01 = 0.0f;
    float tmp_11 = 0.0f;
    float tmp_21 = 0.0f;
    float tmp_31 = 0.0f;
    float tmp_02 = 0.0f;
    float tmp_12 = 0.0f;
    float tmp_22 = 0.0f;
    float tmp_32 = 0.0f;
    float tmp_03 = 0.0f;
    float tmp_13 = 0.0f;
    float tmp_23 = 0.0f;
    float tmp_33 = 0.0f;
    float a_00;
    float a_10;
    float a_20;
    float a_30;
    float b_00;
    float b_01;
    float b_02;
    float b_03;

    for (int subkk = 0; subkk < K; subkk+=16) {
        // 2 warp shuffle
        block_A[block_idx_a] = A[global_idx_a+subkk];
        block_A[block_idx_a+32] = A[global_idx_a+subkk+2*K];
        block_A[block_idx_a+64] = A[global_idx_a+subkk+4*K];
        block_A[block_idx_a+96] = A[global_idx_a+subkk+6*K];
        // block_A[block_idx_a] = A[global_idx_a+subkk];
        // block_A[block_idx_a+32] = A[global_idx_a+subkk+2*K];
        // block_A[block_idx_a+64] = A[global_idx_a+subkk+4*K];
        // block_A[block_idx_a+96] = A[global_idx_a+subkk+6*K];

        block_B[block_idx_b] = B[global_idx_b + subkk*N];
        block_B[block_idx_b+64] = B[global_idx_b + subkk*N + N];
        block_B[block_idx_b+128] = B[global_idx_b + subkk*N + 2*N];
        block_B[block_idx_b+192] = B[global_idx_b + subkk*N + 3*N];


        __syncthreads();
        for (int kk = 0; kk < 16; kk++) {
            b_00 = block_B[block_idx_c_y+kk*64];
            b_01 = block_B[block_idx_c_y+1+kk*64];
            b_02 = block_B[block_idx_c_y+2+kk*64];
            b_03 = block_B[block_idx_c_y+3+kk*64];

            a_00 = block_A[block_idx_c_x*16+kk];
            a_10 = block_A[block_idx_c_x*16+16+kk];
            a_20 = block_A[block_idx_c_x*16+32+kk];
            a_30 = block_A[block_idx_c_x*16+48+kk];
            // a_00 = block_A[block_idx_a_x*16+kk];
            // a_10 = block_A[block_idx_a_x*16+32+kk];
            // a_20 = block_A[block_idx_a_x*16+64+kk];
            // a_30 = block_A[block_idx_a_x*16+96+kk];

            tmp_00 += b_00 * a_00;
            tmp_10 += b_00 * a_10;
            tmp_20 += b_00 * a_20;
            tmp_30 += b_00 * a_30;
            tmp_01 += b_01 * a_00;
            tmp_11 += b_01 * a_10;
            tmp_21 += b_01 * a_20;
            tmp_31 += b_01 * a_30;
            tmp_02 += b_02 * a_00;
            tmp_12 += b_02 * a_10;
            tmp_22 += b_02 * a_20;
            tmp_32 += b_02 * a_30;
            tmp_03 += b_03 * a_00;
            tmp_13 += b_03 * a_10;
            tmp_23 += b_03 * a_20;
            tmp_33 += b_03 * a_30;
        }
        __syncthreads();
    }
    C[global_idx_c] = tmp_00;
    C[global_idx_c+1] = tmp_01;
    C[global_idx_c+2] = tmp_02;
    C[global_idx_c+3] = tmp_03;
    C[global_idx_c+N] = tmp_10;
    C[global_idx_c+N+1] = tmp_11;
    C[global_idx_c+N+2] = tmp_12;
    C[global_idx_c+N+3] = tmp_13;
    C[global_idx_c+2*N] = tmp_20;
    C[global_idx_c+2*N+1] = tmp_21;
    C[global_idx_c+2*N+2] = tmp_22;
    C[global_idx_c+2*N+3] = tmp_23;
    C[global_idx_c+3*N] = tmp_30;
    C[global_idx_c+3*N+1] = tmp_31;
    C[global_idx_c+3*N+2] = tmp_32;
    C[global_idx_c+3*N+3] = tmp_33;
    // if (global_idx_c == 4096*4) {
    //     printf("tmp = %f\n", tmp_00);
    // }
}

float opt2(float *A, float *B, float *C, int iter) {
    float time_elapsed = 0.0;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // for MM = NN = KK = 2048
    dim3 block(TPBX*TPBY);
    dim3 grid(MM/TPBX/4, NN/TPBY/4);
    for (int ii = 0; ii < iter; ii++) {
        opt2_kernel<<<grid, block>>>(A, B, C, MM, KK, NN);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);
    return time_elapsed;
}