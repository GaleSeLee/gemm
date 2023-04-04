// 2060
// 922.221
// 14.2958%
// add task num for each threads x4
#include "gemm.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define TPBX 8
#define TPBY 32

__global__
void opt1_kernel(float *A, float *B, float *C, int M, int K, int N) {
    int idx_x = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = idx_x * NN + idx_y;

    __shared__ float block_A[32][33];
    __shared__ float block_B[32][33];

    float tmp_0 = 0.0f;
    float tmp_1 = 0.0f;
    float tmp_2 = 0.0f;
    float tmp_3 = 0.0f;
    float b00;
    for (int ii = 0; ii < M; ii+=32) {
        // trans A
        block_A[threadIdx.y][threadIdx.x*4] = 
            A[idx_x*K + threadIdx.y+ii];
        block_A[threadIdx.y][threadIdx.x*4+1] = 
            A[idx_x*K + threadIdx.y+ii + K];
        block_A[threadIdx.y][threadIdx.x*4+2] = 
            A[idx_x*K + threadIdx.y+ii + 2 * K];
        block_A[threadIdx.y][threadIdx.x*4+3] = 
            A[idx_x*K + threadIdx.y+ii + 3 * K];
        // No Trans B
        block_B[threadIdx.x*4][threadIdx.y] = 
            B[idx_y + (ii + threadIdx.x*4) * N];
        block_B[threadIdx.x*4+1][threadIdx.y] = 
            B[idx_y + (ii + threadIdx.x*4+1) * N];
        block_B[threadIdx.x*4+2][threadIdx.y] = 
            B[idx_y + (ii + threadIdx.x*4+2) * N];
        block_B[threadIdx.x*4+3][threadIdx.y] = 
            B[idx_y + (ii + threadIdx.x*4+3) * N];
        __syncthreads();
#pragma unroll 32
        for (int kk = 0; kk < 32; kk++) {
            b00 = block_B[kk][threadIdx.y] ;
            tmp_0 += block_A[kk][threadIdx.x*4] * b00;
            tmp_1 += block_A[kk][threadIdx.x*4+1] * b00;
            tmp_2 += block_A[kk][threadIdx.x*4+2] * b00;
            tmp_3 += block_A[kk][threadIdx.x*4+3] * b00;
        }
        __syncthreads();
    }
    C[idx] = tmp_0;
    C[idx+N] = tmp_1;
    C[idx+2*N] = tmp_2;
    C[idx+3*N] = tmp_3;
}

float opt1(float *A, float *B, float *C, int iter) {
    float time_elapsed = 0.0;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // for MM = NN = KK = 2048
    dim3 block(TPBX, TPBY);
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