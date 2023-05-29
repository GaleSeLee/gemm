// 2060
// 4330.5
// 67%-70%
#include "gemm.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define TPBX 16
#define TPBY 16

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__
void opt4_kernel(float *A, float *B, float *C, int M, int K, int N) {
    const int tx = threadIdx.x;
    __shared__ float block_A[1024];
    __shared__ float block_B[1024];
    const int warp_id = tx / 32;
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;
    const int lane_id = tx % 32;
    // for global A index
    const int global_A_row = tx / 2 + blockIdx.x * 128;
    const int global_A_col = tx % 2 * 4; // + kk + sub_kk * 8

    // for global B index
    const int global_B_row = warp_id; // + kk + sub_kk * 8
    const int global_B_col = blockIdx.y * 128 + lane_id * 4;
    
    // for C index
    // if not base, row = base + 16
    //              col = base + 32
    const int global_C_row_base =  blockIdx.x * 128 + warp_row * 32 + lane_id / 8 * 4;
    const int global_C_col_base = blockIdx.y * 128 + warp_col * 64 + lane_id % 8 * 4;
    const int global_C_row_delta = global_C_row_base + 16;
    const int global_C_col_delta = global_C_col_base + 32;

    const int block_A_row_base = global_C_row_base - blockIdx.x * 128;
    const int block_A_row_delta = global_C_row_delta - blockIdx.x * 128;
    const int block_B_col_base = global_C_col_base - blockIdx.y * 128;
    const int block_B_col_delta = global_C_col_delta - blockIdx.y * 128;
    
    float a_frag[2][4];
    float b_frag[2][4];
    float out_frag[2][4][4] = {0};
    for (int ii = 0; ii < K; ii+=8) {
        // read A and B to shared memory
        FETCH_FLOAT4(&block_A[tx*4]) = FETCH_FLOAT4(&A[global_A_row * K + global_A_col + ii]);
        FETCH_FLOAT4(&block_B[tx*4]) = FETCH_FLOAT4(&B[(global_B_row + ii) * N + global_B_col]);
        __syncthreads();
        for (int jj = 0; jj < 8; jj++) {
            // read A and B 8*1 frag to registers
            // outer product and write to c register
            a_frag[0][0] = block_A[block_A_row_base];
            a_frag[0][1] = block_A[block_A_row_base+8];
            a_frag[0][2] = block_A[block_A_row_base+16];
            a_frag[0][3] = block_A[block_A_row_base+24];
            a_frag[1][0] = block_A[block_A_row_delta];
            a_frag[1][1] = block_A[block_A_row_delta+8];
            a_frag[1][2] = block_A[block_A_row_delta+16];
            a_frag[1][3] = block_A[block_A_row_delta+24];
            b_frag[0][0] = block_B[block_B_col_base];
            b_frag[0][1] = block_B[block_B_col_base+1];
            b_frag[0][2] = block_B[block_B_col_base+2];
            b_frag[0][3] = block_B[block_B_col_base+3];
            b_frag[1][0] = block_B[block_B_col_delta];
            b_frag[1][1] = block_B[block_B_col_delta+1];
            b_frag[1][2] = block_B[block_B_col_delta+2];
            b_frag[1][3] = block_B[block_B_col_delta+3];
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