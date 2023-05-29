// 2060
// 4330.5
// 67%-70%
#include "gemm.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define TPBX 16
#define TPBY 16

__global__
void opt4_kernel(float *A, float *B, float *C, int M, int K, int N) {
    const int tx = threadIdx.x;
    __shared__ float block_A[1024];
    __shared__ float block_B[1024];

    // float a_0_0 = 0.0f;
    // float a_0_1 = 0.0f;
    // float a_0_2 = 0.0f;
    // float a_0_3 = 0.0f;
    // float a_1_0 = 0.0f;
    // float a_1_1 = 0.0f;
    // float a_1_2 = 0.0f;
    // float a_1_3 = 0.0f;
    // float b_0_0 = 0.0f;
    // float b_0_1 = 0.0f;
    // float b_0_2 = 0.0f;
    // float b_0_3 = 0.0f;
    // float b_1_0 = 0.0f;
    // float b_1_1 = 0.0f;
    // float b_1_2 = 0.0f;
    // float b_1_3 = 0.0f;
    float a_frag[2][4];
    float b_frag[2][4];
    float out_frag[2][4][4];
    for (int ii = 0; ii < K; ii+=8) {
        // read A and B to shared memory
        
        for (int jj = 0; jj < 8; jj++) {
            // read A and B 8*1 frag to registers
            // outer product and write to c register
        }

    }

    // write the c register to variable
    
}

float opt4(float *A, float *B, float *C, int iter) {
    float time_elapsed = 0.0;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // for MM = NN = KK = 2048
    dim3 block(TPBX*TPBY);
    dim3 grid(MM/TPBX/8, NN/TPBY/8);
    for (int ii = 0; ii < iter; ii++) {
        opt4_kernel<<<grid, block>>>(A, B, C, MM, KK, NN);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);
    return time_elapsed;
}