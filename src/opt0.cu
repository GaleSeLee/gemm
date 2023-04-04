#include "gemm.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define TPBX 16
#define TPBY 16

__global__
void baseline_kernel(float *A, float *B, float *C, int M, int K, int N) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = idx_x * NN + idx_y;

    C[idx] = 0;
    for (int kk = 0; kk < K; kk++) {
        C[idx] += A[idx_x*K + kk] * B[idx_y + kk*N];
    }
}

float baseline(float *A, float *B, float *C, int iter) {
    float time_elapsed = 0.0;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // for MM = NN = KK = 2048
    dim3 block(TPBX, TPBY);
    dim3 grid(MM/TPBX, NN/TPBY);
    for (int ii = 0; ii < iter; ii++) {
        baseline_kernel<<<grid, block>>>(A, B, C, MM, KK, NN);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);
    return time_elapsed;
}