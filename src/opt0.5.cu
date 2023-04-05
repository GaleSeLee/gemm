// 2060
// 967.66
// 15%
#include "gemm.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define TPBX 32
#define TPBY 32

__global__
void opt05_kernel(float *A, float *B, float *C, int M, int K, int N) {
    const int tx = threadIdx.x;
    const int block_idx_x = tx >> 5;
    const int block_idx_y = tx & 31; // lane
    const int global_idx_x = block_idx_x + blockIdx.x * 32;
    const int global_idx_y = block_idx_y + blockIdx.y * 32;
    const int global_idx_c = global_idx_y + global_idx_x * N;
    __shared__ float block_A[1024];
    __shared__ float block_B[1024];
    
    float tmp = 0.0f;

    for (int ii = 0; ii < M; ii+=32) {
        block_A[block_idx_x*32+block_idx_y] = A[global_idx_x * K + block_idx_y + ii];
        block_B[block_idx_x*32+block_idx_y] = B[global_idx_y + (block_idx_x + ii) * N];
        __syncthreads();
        for (int kk = 0; kk < 32; kk++) {
            tmp +=  block_A[block_idx_x*32+kk] * block_B[kk*32+block_idx_y];
        }
        __syncthreads();
    }
    C[global_idx_c] = tmp;
}

float opt05(float *A, float *B, float *C, int iter) {
    float time_elapsed = 0.0;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // for MM = NN = KK = 2048
    dim3 block(TPBX*TPBY);
    dim3 grid(MM/TPBX, NN/TPBY);
    for (int ii = 0; ii < iter; ii++) {
        opt05_kernel<<<grid, block>>>(A, B, C, MM, KK, NN);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);
    return time_elapsed;
}