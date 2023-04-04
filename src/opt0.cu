// 2060
// 417.715
// 6.47519
#include "gemm.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define TPBX 32
#define TPBY 32

__global__
void opt0_kernel(float *A, float *B, float *C, int M, int K, int N) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = idx_x * NN + idx_y;

    // 33 for reducing bank conflict
    __shared__ float block_A[32][33];
    __shared__ float block_B[32][33];

    float tmp = 0.0f;
    for (int ii = 0; ii < M; ii+=32) {
        // trans A
        block_A[threadIdx.y][threadIdx.x] = 
            A[idx_x*K + threadIdx.y+ii];
        // No Trans B
        block_B[threadIdx.x][threadIdx.y] = 
            B[idx_y + (ii + threadIdx.x) * N];
        __syncthreads();
        for (int kk = 0; kk < 32; kk++) {
            tmp += block_A[kk][threadIdx.x] * block_B[kk][threadIdx.y];
        }
        __syncthreads();
    }
    C[idx] = tmp;
}

float opt0(float *A, float *B, float *C, int iter) {
    float time_elapsed = 0.0;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // for MM = NN = KK = 2048
    dim3 block(TPBX, TPBY);
    dim3 grid(MM/TPBX, NN/TPBY);
    for (int ii = 0; ii < iter; ii++) {
        opt0_kernel<<<grid, block>>>(A, B, C, MM, KK, NN);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);
    return time_elapsed;
}