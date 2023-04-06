// 2060
// 4330.5
// 67%
#include "gemm.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define TPBX 16
#define TPBY 16

__global__
void opt3_kernel(float *A, float *B, float *C, int M, int K, int N) {
    const int tx = threadIdx.x;
    // A 128 x 8
    const int block_idx_a_y = tx & 7;
    const int block_idx_a_x = ((tx>>5)<<4) + ((tx&31)>>3);
    const int block_idx_a = block_idx_a_x * 8 + block_idx_a_y;
    const int global_idx_a_x = block_idx_a_x + blockIdx.x * 128;
    const int global_idx_a_y = block_idx_a_y + blockIdx.y * 8;
    const int global_idx_a = global_idx_a_x * K + global_idx_a_y;

    // store idx
    const int block_idx_b_y = tx & 127;
    const int block_idx_b_x = tx >> 7 << 2;
    const int block_idx_b = block_idx_b_x * 128 + block_idx_b_y;
    // load idx
    const int global_idx_b_x = block_idx_b_x + blockIdx.x * 8;
    const int global_idx_b_y = block_idx_b_y + blockIdx.y * 128;
    const int global_idx_b = global_idx_b_x * N + global_idx_b_y;

    const int block_idx_c_y = ((tx & 15) << 3);
    const int block_idx_c_x = ((tx >> 4) << 3);
    const int global_idx_c_x = block_idx_c_x + blockIdx.x * 128;
    const int global_idx_c_y = block_idx_c_y + blockIdx.y * 128;
    const int global_idx_c = global_idx_c_x * N + global_idx_c_y;
    __shared__ float block_A[1024];
    __shared__ float block_B[1024];
    
    float tmp_00 = 0.0f;
    float tmp_01 = 0.0f;
    float tmp_02 = 0.0f;
    float tmp_03 = 0.0f;
    float tmp_04 = 0.0f;
    float tmp_05 = 0.0f;
    float tmp_06 = 0.0f;
    float tmp_07 = 0.0f;
    float tmp_10 = 0.0f;
    float tmp_11 = 0.0f;
    float tmp_12 = 0.0f;
    float tmp_13 = 0.0f;
    float tmp_14 = 0.0f;
    float tmp_15 = 0.0f;
    float tmp_16 = 0.0f;
    float tmp_17 = 0.0f;
    float tmp_20 = 0.0f;
    float tmp_21 = 0.0f;
    float tmp_22 = 0.0f;
    float tmp_23 = 0.0f;
    float tmp_24 = 0.0f;
    float tmp_25 = 0.0f;
    float tmp_26 = 0.0f;
    float tmp_27 = 0.0f;
    float tmp_30 = 0.0f;
    float tmp_31 = 0.0f;
    float tmp_32 = 0.0f;
    float tmp_33 = 0.0f;
    float tmp_34 = 0.0f;
    float tmp_35 = 0.0f;
    float tmp_36 = 0.0f;
    float tmp_37 = 0.0f;
    float tmp_40 = 0.0f;
    float tmp_41 = 0.0f;
    float tmp_42 = 0.0f;
    float tmp_43 = 0.0f;
    float tmp_44 = 0.0f;
    float tmp_45 = 0.0f;
    float tmp_46 = 0.0f;
    float tmp_47 = 0.0f;
    float tmp_50 = 0.0f;
    float tmp_51 = 0.0f;
    float tmp_52 = 0.0f;
    float tmp_53 = 0.0f;
    float tmp_54 = 0.0f;
    float tmp_55 = 0.0f;
    float tmp_56 = 0.0f;
    float tmp_57 = 0.0f;
    float tmp_60 = 0.0f;
    float tmp_61 = 0.0f;
    float tmp_62 = 0.0f;
    float tmp_63 = 0.0f;
    float tmp_64 = 0.0f;
    float tmp_65 = 0.0f;
    float tmp_66 = 0.0f;
    float tmp_67 = 0.0f;
    float tmp_70 = 0.0f;
    float tmp_71 = 0.0f;
    float tmp_72 = 0.0f;
    float tmp_73 = 0.0f;
    float tmp_74 = 0.0f;
    float tmp_75 = 0.0f;
    float tmp_76 = 0.0f;
    float tmp_77 = 0.0f;

    float a_00;
    float a_10;
    float a_20;
    float a_30;
    float a_40;
    float a_50;
    float a_60;
    float a_70;
    float b_00;
    float b_01;
    float b_02;
    float b_03;
    float b_04;
    float b_05;
    float b_06;
    float b_07;

    for (int subkk = 0; subkk < K; subkk+=8) {
        // 2 warp shuffle
        block_A[block_idx_a] = A[global_idx_a+subkk];
        block_A[block_idx_a+32] = A[global_idx_a+subkk+4*K];
        block_A[block_idx_a+64] = A[global_idx_a+subkk+8*K];
        block_A[block_idx_a+96] = A[global_idx_a+subkk+12*K];
        // block_A[block_idx_a] = A[global_idx_a+subkk];
        // block_A[block_idx_a+32] = A[global_idx_a+subkk+2*K];
        // block_A[block_idx_a+64] = A[global_idx_a+subkk+4*K];
        // block_A[block_idx_a+96] = A[global_idx_a+subkk+6*K];

        block_B[block_idx_b] = B[global_idx_b + subkk*N];
        block_B[block_idx_b+128] = B[global_idx_b + subkk*N + N];
        block_B[block_idx_b+256] = B[global_idx_b + subkk*N + 2*N];
        block_B[block_idx_b+384] = B[global_idx_b + subkk*N + 3*N];


        __syncthreads();
        for (int kk = 0; kk < 8; kk++) {
            int idx_b = block_idx_c_y + kk * 128;
            b_00 = block_B[idx_b];
            b_01 = block_B[idx_b+1];
            b_02 = block_B[idx_b+2];
            b_03 = block_B[idx_b+3];
            b_04 = block_B[idx_b+4];
            b_05 = block_B[idx_b+5];
            b_06 = block_B[idx_b+6];
            b_07 = block_B[idx_b+7];
            // b_00 = block_B[0];
            // b_01 = block_B[1];
            // b_02 = block_B[2];
            // b_03 = block_B[3];
            // b_04 = block_B[4];
            // b_05 = block_B[5];
            // b_06 = block_B[6];
            // b_07 = block_B[7];

            // a_00 = block_A[0];
            // a_10 = block_A[0];
            // a_20 = block_A[0];
            // a_30 = block_A[0];
            // a_40 = block_A[0];
            // a_50 = block_A[0];
            // a_60 = block_A[0];
            // a_70 = block_A[0];
            a_00 = block_A[block_idx_c_x*8+kk];
            a_10 = block_A[block_idx_c_x*8+8+kk];
            a_20 = block_A[block_idx_c_x*8+16+kk];
            a_30 = block_A[block_idx_c_x*8+24+kk];
            a_40 = block_A[block_idx_c_x*8+32+kk];
            a_50 = block_A[block_idx_c_x*8+40+kk];
            a_60 = block_A[block_idx_c_x*8+48+kk];
            a_70 = block_A[block_idx_c_x*8+56+kk];

            // a_00 = block_A[block_idx_a_x*16+kk];
            // a_10 = block_A[block_idx_a_x*16+32+kk];
            // a_20 = block_A[block_idx_a_x*16+64+kk];
            // a_30 = block_A[block_idx_a_x*16+96+kk];

            tmp_00 += a_00 * b_00;
            tmp_01 += a_00 * b_01;
            tmp_02 += a_00 * b_02;
            tmp_03 += a_00 * b_03;
            tmp_04 += a_00 * b_04;
            tmp_05 += a_00 * b_05;
            tmp_06 += a_00 * b_06;
            tmp_07 += a_00 * b_07;
            tmp_10 += a_10 * b_00;
            tmp_11 += a_10 * b_01;
            tmp_12 += a_10 * b_02;
            tmp_13 += a_10 * b_03;
            tmp_14 += a_10 * b_04;
            tmp_15 += a_10 * b_05;
            tmp_16 += a_10 * b_06;
            tmp_17 += a_10 * b_07;
            tmp_20 += a_20 * b_00;
            tmp_21 += a_20 * b_01;
            tmp_22 += a_20 * b_02;
            tmp_23 += a_20 * b_03;
            tmp_24 += a_20 * b_04;
            tmp_25 += a_20 * b_05;
            tmp_26 += a_20 * b_06;
            tmp_27 += a_20 * b_07;
            tmp_30 += a_30 * b_00;
            tmp_31 += a_30 * b_01;
            tmp_32 += a_30 * b_02;
            tmp_33 += a_30 * b_03;
            tmp_34 += a_30 * b_04;
            tmp_35 += a_30 * b_05;
            tmp_36 += a_30 * b_06;
            tmp_37 += a_30 * b_07;
            tmp_40 += a_40 * b_00;
            tmp_41 += a_40 * b_01;
            tmp_42 += a_40 * b_02;
            tmp_43 += a_40 * b_03;
            tmp_44 += a_40 * b_04;
            tmp_45 += a_40 * b_05;
            tmp_46 += a_40 * b_06;
            tmp_47 += a_40 * b_07;
            tmp_50 += a_50 * b_00;
            tmp_51 += a_50 * b_01;
            tmp_52 += a_50 * b_02;
            tmp_53 += a_50 * b_03;
            tmp_54 += a_50 * b_04;
            tmp_55 += a_50 * b_05;
            tmp_56 += a_50 * b_06;
            tmp_57 += a_50 * b_07;
            tmp_60 += a_60 * b_00;
            tmp_61 += a_60 * b_01;
            tmp_62 += a_60 * b_02;
            tmp_63 += a_60 * b_03;
            tmp_64 += a_60 * b_04;
            tmp_65 += a_60 * b_05;
            tmp_66 += a_60 * b_06;
            tmp_67 += a_60 * b_07;
            tmp_70 += a_70 * b_00;
            tmp_71 += a_70 * b_01;
            tmp_72 += a_70 * b_02;
            tmp_73 += a_70 * b_03;
            tmp_74 += a_70 * b_04;
            tmp_75 += a_70 * b_05;
            tmp_76 += a_70 * b_06;
            tmp_77 += a_70 * b_07;
        }
        __syncthreads();
    }
    C[global_idx_c+0*N+0] = tmp_00;
    C[global_idx_c+0*N+1] = tmp_01;
    C[global_idx_c+0*N+2] = tmp_02;
    C[global_idx_c+0*N+3] = tmp_03;
    C[global_idx_c+0*N+4] = tmp_04;
    C[global_idx_c+0*N+5] = tmp_05;
    C[global_idx_c+0*N+6] = tmp_06;
    C[global_idx_c+0*N+7] = tmp_07;
    C[global_idx_c+1*N+0] = tmp_10;
    C[global_idx_c+1*N+1] = tmp_11;
    C[global_idx_c+1*N+2] = tmp_12;
    C[global_idx_c+1*N+3] = tmp_13;
    C[global_idx_c+1*N+4] = tmp_14;
    C[global_idx_c+1*N+5] = tmp_15;
    C[global_idx_c+1*N+6] = tmp_16;
    C[global_idx_c+1*N+7] = tmp_17;
    C[global_idx_c+2*N+0] = tmp_20;
    C[global_idx_c+2*N+1] = tmp_21;
    C[global_idx_c+2*N+2] = tmp_22;
    C[global_idx_c+2*N+3] = tmp_23;
    C[global_idx_c+2*N+4] = tmp_24;
    C[global_idx_c+2*N+5] = tmp_25;
    C[global_idx_c+2*N+6] = tmp_26;
    C[global_idx_c+2*N+7] = tmp_27;
    C[global_idx_c+3*N+0] = tmp_30;
    C[global_idx_c+3*N+1] = tmp_31;
    C[global_idx_c+3*N+2] = tmp_32;
    C[global_idx_c+3*N+3] = tmp_33;
    C[global_idx_c+3*N+4] = tmp_34;
    C[global_idx_c+3*N+5] = tmp_35;
    C[global_idx_c+3*N+6] = tmp_36;
    C[global_idx_c+3*N+7] = tmp_37;
    C[global_idx_c+4*N+0] = tmp_40;
    C[global_idx_c+4*N+1] = tmp_41;
    C[global_idx_c+4*N+2] = tmp_42;
    C[global_idx_c+4*N+3] = tmp_43;
    C[global_idx_c+4*N+4] = tmp_44;
    C[global_idx_c+4*N+5] = tmp_45;
    C[global_idx_c+4*N+6] = tmp_46;
    C[global_idx_c+4*N+7] = tmp_47;
    C[global_idx_c+5*N+0] = tmp_50;
    C[global_idx_c+5*N+1] = tmp_51;
    C[global_idx_c+5*N+2] = tmp_52;
    C[global_idx_c+5*N+3] = tmp_53;
    C[global_idx_c+5*N+4] = tmp_54;
    C[global_idx_c+5*N+5] = tmp_55;
    C[global_idx_c+5*N+6] = tmp_56;
    C[global_idx_c+5*N+7] = tmp_57;
    C[global_idx_c+6*N+0] = tmp_60;
    C[global_idx_c+6*N+1] = tmp_61;
    C[global_idx_c+6*N+2] = tmp_62;
    C[global_idx_c+6*N+3] = tmp_63;
    C[global_idx_c+6*N+4] = tmp_64;
    C[global_idx_c+6*N+5] = tmp_65;
    C[global_idx_c+6*N+6] = tmp_66;
    C[global_idx_c+6*N+7] = tmp_67;
    C[global_idx_c+7*N+0] = tmp_70;
    C[global_idx_c+7*N+1] = tmp_71;
    C[global_idx_c+7*N+2] = tmp_72;
    C[global_idx_c+7*N+3] = tmp_73;
    C[global_idx_c+7*N+4] = tmp_74;
    C[global_idx_c+7*N+5] = tmp_75;
    C[global_idx_c+7*N+6] = tmp_76;
    C[global_idx_c+7*N+7] = tmp_77;
}

float opt3(float *A, float *B, float *C, int iter) {
    float time_elapsed = 0.0;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // for MM = NN = KK = 2048
    dim3 block(TPBX*TPBY);
    dim3 grid(MM/TPBX/8, NN/TPBY/8);
    for (int ii = 0; ii < iter; ii++) {
        opt3_kernel<<<grid, block>>>(A, B, C, MM, KK, NN);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);
    return time_elapsed;
}