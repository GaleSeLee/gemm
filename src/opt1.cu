// 2060
// 1989.72
// 30.84%
#include "gemm.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define TPBX 8
#define TPBY 32

__global__
void opt1_kernel(float *A, float *B, float *C, int M, int K, int N) {
    const int tx = threadIdx.x;
    const int block_idx_x_0 = (tx >> 5)*4;
    const int block_idx_x_1 = block_idx_x_0+1;
    const int block_idx_x_2 = block_idx_x_0+2;
    const int block_idx_x_3 = block_idx_x_0+3;
    const int block_idx_y = tx & 31; // lane
    const int global_idx_x_0 = block_idx_x_0 + blockIdx.x * 32;
    const int global_idx_x_1 = block_idx_x_1 + blockIdx.x * 32;
    const int global_idx_x_2 = block_idx_x_2 + blockIdx.x * 32;
    const int global_idx_x_3 = block_idx_x_3 + blockIdx.x * 32;
    const int global_idx_y = block_idx_y + blockIdx.y * 32;
    const int global_idx_c = global_idx_y + global_idx_x_0 * N;
    __shared__ float block_A[1024];
    __shared__ float block_B[1024];
    
    float tmp_00 = 0.0f;
    float tmp_10 = 0.0f;
    float tmp_20 = 0.0f;
    float tmp_30 = 0.0f;

    for (int ii = 0; ii < M; ii+=32) {
        block_A[block_idx_x_0*32+block_idx_y] = A[global_idx_x_0 * K + block_idx_y + ii];
        block_A[block_idx_x_1*32+block_idx_y] = A[global_idx_x_1 * K + block_idx_y + ii];
        block_A[block_idx_x_2*32+block_idx_y] = A[global_idx_x_2 * K + block_idx_y + ii];
        block_A[block_idx_x_3*32+block_idx_y] = A[global_idx_x_3 * K + block_idx_y + ii];
        // tmp_00= A[global_idx_x_0 * K + block_idx_y + ii];
        // tmp_10= A[global_idx_x_1 * K + block_idx_y + ii];
        // tmp_20= A[global_idx_x_2 * K + block_idx_y + ii];
        // tmp_30= A[global_idx_x_3 * K + block_idx_y + ii];
        
        block_B[block_idx_x_0*32+block_idx_y] = B[global_idx_y + (block_idx_x_0 + ii) * N];
        block_B[block_idx_x_1*32+block_idx_y] = B[global_idx_y + (block_idx_x_1 + ii) * N];
        block_B[block_idx_x_2*32+block_idx_y] = B[global_idx_y + (block_idx_x_2 + ii) * N];
        block_B[block_idx_x_3*32+block_idx_y] = B[global_idx_y + (block_idx_x_3 + ii) * N];
        __syncthreads();
        for (int kk = 0; kk < 32; kk++) {
            tmp_00 +=  block_A[block_idx_x_0*32+kk] * block_B[kk*32+block_idx_y];
            tmp_10 +=  block_A[block_idx_x_1*32+kk] * block_B[kk*32+block_idx_y];
            tmp_20 +=  block_A[block_idx_x_2*32+kk] * block_B[kk*32+block_idx_y];
            tmp_30 +=  block_A[block_idx_x_3*32+kk] * block_B[kk*32+block_idx_y];
        } 
        __syncthreads();
        // block_A[block_idx_x_0+32*block_idx_y] = A[global_idx_x_0 * K + block_idx_y + ii];
        // block_A[block_idx_x_1+32*block_idx_y] = A[global_idx_x_1 * K + block_idx_y + ii];
        // block_A[block_idx_x_2+32*block_idx_y] = A[global_idx_x_2 * K + block_idx_y + ii];
        // block_A[block_idx_x_3+32*block_idx_y] = A[global_idx_x_3 * K + block_idx_y + ii];

        // block_B[block_idx_x_0*32+block_idx_y] = B[global_idx_y + (block_idx_x_0 + ii) * N];
        // block_B[block_idx_x_1*32+block_idx_y] = B[global_idx_y + (block_idx_x_1 + ii) * N];
        // block_B[block_idx_x_2*32+block_idx_y] = B[global_idx_y + (block_idx_x_2 + ii) * N];
        // block_B[block_idx_x_3*32+block_idx_y] = B[global_idx_y + (block_idx_x_3 + ii) * N];
        // __syncthreads();
        // for (int kk = 0; kk < 32; kk++) {
        //     tmp_00 +=  block_A[block_idx_x_0+32*kk] * block_B[kk*32+block_idx_y];
        //     tmp_10 +=  block_A[block_idx_x_1+32*kk] * block_B[kk*32+block_idx_y];
        //     tmp_20 +=  block_A[block_idx_x_2+32*kk] * block_B[kk*32+block_idx_y];
        //     tmp_30 +=  block_A[block_idx_x_3+32*kk] * block_B[kk*32+block_idx_y];
        // } 
        // __syncthreads();
    }
    C[global_idx_c] = tmp_00;
    C[global_idx_c+N] = tmp_10;
    C[global_idx_c+N+N] = tmp_20;
    C[global_idx_c+N+N+N] = tmp_30;
}

float opt1(float *A, float *B, float *C, int iter) {
    float time_elapsed = 0.0;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // for MM = NN = KK = 2048
    dim3 block(TPBX*TPBY);
    dim3 grid(MM/TPBX/4, NN/TPBY);
    for (int ii = 0; ii < iter; ii++) {
        opt1_kernel<<<grid, block>>>(A, B, C, MM, KK, NN);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);
    return time_elapsed;
}