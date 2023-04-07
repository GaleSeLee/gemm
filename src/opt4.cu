#include "gemm.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define TPBX 16
#define TPBY 16

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

// K: ldA
// N: ldB

#define BLOCK_SIZE_M  32  // height of block of C that each thread block calculate
#define BLOCK_SIZE_K  32  // width of block of A that each thread block load into shared memory
#define BLOCK_SIZE_N  32  // width of block of C that each thread block calculate
#define THREAD_SIZE_Y  2 // height of block of C that each thread calculate
#define THREAD_SIZE_X  2  // width of block of C that each thread calculate

__global__ void opt4_kernel( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C, 
    const int M,
    const int K,
    const int N) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // the threads number in Block of X,Y
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    // thread id in cur Block
    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    // shared memory
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // registers for A and B
    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];
    // registers load global memory
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);
    float ldg_a_reg[4*ldg_num_a];
    float ldg_b_reg[4*ldg_num_b];

    // threads number in one row
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4; 
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    A = &A[(BLOCK_SIZE_M * by)* K];
    B = &B[BLOCK_SIZE_N * bx];

    //transfer first tile from global mem to shared mem
    // load A from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
            A_TILE_ROW_START + i, // row
            A_TILE_COL, // col
            K )]);
        As[0][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
        As[0][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
        As[0][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
        As[0][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
    }
    // load B from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                B_TILE_ROW_START + i, // row
                B_TILE_COL, // col
                N )]);
    }
    __syncthreads();
    // load A from shared memory to register
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
    }
    // load B from shared memory to register
    #pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
    }

    int write_stage_idx = 1;
    int tile_idx = 0;
    do{
        tile_idx += BLOCK_SIZE_K;
        // load next tile from global mem
        if(tile_idx< K){
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                    A_TILE_ROW_START + i, // row
                    A_TILE_COL + tile_idx, // col
                    K )]);
            }
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
                    tile_idx + B_TILE_ROW_START + i, // row
                    B_TILE_COL, // col
                    N )]);
            }
        }

        int load_stage_idx = write_stage_idx ^ 1;

        #pragma unroll
        for(int j=0; j<BLOCK_SIZE_K-1; ++j){
            // load next tile from shared mem to register 
            // load A from shared memory to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
                FETCH_FLOAT4(frag_a[(j+1)%2][thread_y]) = FETCH_FLOAT4(As[load_stage_idx][j+1][THREAD_SIZE_Y * ty + thread_y]);
            }
            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                FETCH_FLOAT4(frag_b[(j+1)%2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx][j+1][THREAD_SIZE_X * tx + thread_x]);
            }
            // compute C THREAD_SIZE_X x THREAD_SIZE_Y
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j%2][thread_y] * frag_b[j%2][thread_x];
                }
            }
        }

        if(tile_idx < K){
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
                As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
                As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
                As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
            }
            // load B from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            // use double buffer, only need one sync
            __syncthreads();
            // switch
            write_stage_idx ^= 1;
        }

        // load first tile from shared mem to register of next iter
        // load A from shared memory to register
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
            FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[load_stage_idx^1][0][THREAD_SIZE_Y * ty + thread_y]);
        }
        // load B from shared memory to register
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][THREAD_SIZE_X * tx + thread_x]);
        }
        //compute last tile mma THREAD_SIZE_X x THREAD_SIZE_Y
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    }while(tile_idx< K);

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x+=4) {
            FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
        }
    }
}



// #include<stdio.h>
// #include<stdlib.h>
// #define A(i,j) A[(i) + (j)*lda]
// #define B(i,j) B[(i) + (j)*ldb]
// #define C(i,j) C[(i) + (j)*ldc]
// #define sa8(i,j) sa8[((j)<<7) + (i)]
// #define sb8(i,j) sb8[((j)<<7) + (i)]
// #define MS_8 128
// #define NS_8 128
// #define KS_8 8
// //v1 += v2 * s3, vector scaling
// #define vscal(v1, v2, s3)\
//     v1.x+=v2.x*s3;\
//     v1.y+=v2.y*s3;\
//     v1.z+=v2.z*s3;\
//     v1.w+=v2.w*s3;
// //v1 = alpha * v2 + beta * v3, simd fma
// #define simd_axpby(v1, alpha, v2, beta, v3)\
//     v1.x=alpha*v2.x+beta*v3.x;\
//     v1.y=alpha*v2.y+beta*v3.y;\
//     v1.z=alpha*v2.z+beta*v3.z;\
//     v1.w=alpha*v2.w+beta*v3.w;
// #define vload(v1,addr)\
//     v1 = *((float4 *)(addr));
// #define vstore(addr,v1)\
//     *((float4 *)(addr)) = v1;
// // cache blocking version, without register-level data re-use
// // with memory coelascing on shared memory
// // more workloads per thread. 8x8 micro kernel.
// // adopt vetorized load/store
// __global__  __launch_bounds__(256)
// void mysgemm_v8(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
//     int lda = M, ldb = K, ldc = M;
//     int tx = threadIdx.x;
//     int bx = blockIdx.x, by = blockIdx.y;
//     int row_a = (tx&31)<<2, col_a = tx>>5;
//     int row_b = (tx&1)<<2, col_b = tx>>1;
//     int lda8 = lda<<3;
//     int row_c = (tx&15)<<3, col_c = (tx>>4)<<3;
//     A = &A((bx<<7),0);
//     B = &B(0,(by<<7));
//     C = &C((bx<<7),(by<<7));//the TB size is 128.
//     __shared__ float sa8[1024];
//     __shared__ float sb8[1024];
//     float4 Av1, Av2, Bv1, Bv2, Cv[16], Cres[16];
//     memset(Cres, 0, sizeof(Cres));//clear registers
//     for (int k_count = 0; k_count<K; k_count+=KS_8){
//         vload(Av1, &A(row_a,col_a))
//         vload(Bv1, &B(row_b,col_b))
//         ((float4 *)sa8)[tx] = Av1;
//         sb8(col_b,row_b)=Bv1.x;
//         sb8(col_b,row_b+1)=Bv1.y;
//         sb8(col_b,row_b+2)=Bv1.z;
//         sb8(col_b,row_b+3)=Bv1.w;
//         A+=lda8;B+=8;
//         __syncthreads();
//         #pragma unroll
//         for (int inner_k_count=0;inner_k_count<KS_8;inner_k_count++){
//             vload(Av1, &sa8(row_c,inner_k_count))
//             vload(Av2, &sa8(row_c+4,inner_k_count))
//             vload(Bv1, &sb8(col_c,inner_k_count))
//             vload(Bv2, &sb8(col_c+4,inner_k_count))
//             vscal(Cres[0], Av1, Bv1.x)
//             vscal(Cres[1], Av2, Bv1.x)
//             vscal(Cres[2], Av1, Bv1.y)
//             vscal(Cres[3], Av2, Bv1.y)
//             vscal(Cres[4], Av1, Bv1.z)
//             vscal(Cres[5], Av2, Bv1.z)
//             vscal(Cres[6], Av1, Bv1.w)
//             vscal(Cres[7], Av2, Bv1.w)
//             vscal(Cres[8], Av1, Bv2.x)
//             vscal(Cres[9], Av2, Bv2.x)
//             vscal(Cres[10], Av1, Bv2.y)
//             vscal(Cres[11], Av2, Bv2.y)
//             vscal(Cres[12], Av1, Bv2.z)
//             vscal(Cres[13], Av2, Bv2.z)
//             vscal(Cres[14], Av1, Bv2.w)
//             vscal(Cres[15], Av2, Bv2.w)
//         }
//         __syncthreads();
//     }
//     vload(Cv[0], &C(row_c,col_c))
//     vload(Cv[1], &C(row_c+4,col_c))
//     vload(Cv[2], &C(row_c,col_c+1))
//     vload(Cv[3], &C(row_c+4,col_c+1))
//     vload(Cv[4], &C(row_c,col_c+2))
//     vload(Cv[5], &C(row_c+4,col_c+2))
//     vload(Cv[6], &C(row_c,col_c+3))
//     vload(Cv[7], &C(row_c+4,col_c+3))
//     vload(Cv[8], &C(row_c,col_c+4))
//     vload(Cv[9], &C(row_c+4,col_c+4))
//     vload(Cv[10], &C(row_c,col_c+5))
//     vload(Cv[11], &C(row_c+4,col_c+5))
//     vload(Cv[12], &C(row_c,col_c+6))
//     vload(Cv[13], &C(row_c+4,col_c+6))
//     vload(Cv[14], &C(row_c,col_c+7))
//     vload(Cv[15], &C(row_c+4,col_c+7))
    
//     simd_axpby(Cres[0],alpha,Cres[0],beta,Cv[0])
//     simd_axpby(Cres[1],alpha,Cres[1],beta,Cv[1])
//     simd_axpby(Cres[2],alpha,Cres[2],beta,Cv[2])
//     simd_axpby(Cres[3],alpha,Cres[3],beta,Cv[3])

//     simd_axpby(Cres[4],alpha,Cres[4],beta,Cv[4])
//     simd_axpby(Cres[5],alpha,Cres[5],beta,Cv[5])
//     simd_axpby(Cres[6],alpha,Cres[6],beta,Cv[6])
//     simd_axpby(Cres[7],alpha,Cres[7],beta,Cv[7])

//     simd_axpby(Cres[8],alpha,Cres[8],beta,Cv[8])
//     simd_axpby(Cres[9],alpha,Cres[9],beta,Cv[9])
//     simd_axpby(Cres[10],alpha,Cres[10],beta,Cv[10])
//     simd_axpby(Cres[11],alpha,Cres[11],beta,Cv[11])

//     simd_axpby(Cres[12],alpha,Cres[12],beta,Cv[12])
//     simd_axpby(Cres[13],alpha,Cres[13],beta,Cv[13])
//     simd_axpby(Cres[14],alpha,Cres[14],beta,Cv[14])
//     simd_axpby(Cres[15],alpha,Cres[15],beta,Cv[15])

//     vstore(&C(row_c,col_c), Cres[0])
//     vstore(&C(row_c+4,col_c), Cres[1])
//     vstore(&C(row_c,col_c+1), Cres[2])
//     vstore(&C(row_c+4,col_c+1), Cres[3])
//     vstore(&C(row_c,col_c+2), Cres[4])
//     vstore(&C(row_c+4,col_c+2), Cres[5])
//     vstore(&C(row_c,col_c+3), Cres[6])
//     vstore(&C(row_c+4,col_c+3), Cres[7])
//     vstore(&C(row_c,col_c+4), Cres[8])
//     vstore(&C(row_c+4,col_c+4), Cres[9])
//     vstore(&C(row_c,col_c+5), Cres[10])
//     vstore(&C(row_c+4,col_c+5), Cres[11])
//     vstore(&C(row_c,col_c+6), Cres[12])
//     vstore(&C(row_c+4,col_c+6), Cres[13])
//     vstore(&C(row_c,col_c+7), Cres[14])
//     vstore(&C(row_c+4,col_c+7), Cres[15])
// }

#include<stdio.h>
#include<stdlib.h>
#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa9(i,j) sa9[((j)<<7) + (i)]
#define sb9(i,j) sb9[((j)<<7) + (i)]
#define MS_9 128
#define NS_9 128
#define KS_9 8
//v1 += v2 * s3, vector scaling
#define vscal(v1, v2, s3)\
    v1.x+=v2.x*s3;\
    v1.y+=v2.y*s3;\
    v1.z+=v2.z*s3;\
    v1.w+=v2.w*s3;
//v1 = alpha * v2 + beta * v3, simd fma
#define simd_axpby(v1, alpha, v2, beta, v3)\
    v1.x=alpha*v2.x+beta*v3.x;\
    v1.y=alpha*v2.y+beta*v3.y;\
    v1.z=alpha*v2.z+beta*v3.z;\
    v1.w=alpha*v2.w+beta*v3.w;
#define vload(v1,addr)\
    v1 = *((float4 *)(addr));
#define vstore(addr,v1)\
    *((float4 *)(addr)) = v1;
// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 8x8 micro kernel.
// adopt vetorized load/store
__global__  __launch_bounds__(256)
void mysgemm_v9(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int warp_id = tx>>5;
    int lane_id = tx&31;
    int warp_row = warp_id & 3, warp_col = warp_id >> 2;
    int row_w = lane_id&3, col_w = lane_id>>2;
    int row_b = (tx&1)<<2, col_b = tx>>1;
    int lda8 = lda<<3;
    int row_c = (warp_row<<5) + (row_w<<3), col_c = (warp_col<<6) + (col_w<<3);
    int row_a = (tx&31)<<2, col_a = tx>>5;
    A = &A((bx<<7),0);
    B = &B(0,(by<<7));
    C = &C((bx<<7),(by<<7));//the TB size is 128.
    __shared__ float sa9[1024];
    __shared__ float sb9[1024];
    float4 Av1, Av2, Bv1, Bv2, Cv[16], Cres[16];
    memset(Cres, 0, sizeof(Cres));//clear registers
    for (int k_count = 0; k_count<K; k_count+=KS_9){
        /*packing A and B into shared memory*/
        vload(Av1, &A(row_a,col_a))
        vload(Bv1, &B(row_b,col_b))
        ((float4 *)sa9)[tx] = Av1;
        sb9(col_b,row_b)=Bv1.x;
        sb9(col_b,row_b+1)=Bv1.y;
        sb9(col_b,row_b+2)=Bv1.z;
        sb9(col_b,row_b+3)=Bv1.w;
        A+=lda8;B+=8;
        __syncthreads();
        #pragma unroll
        for (int inner_k_count=0;inner_k_count<KS_9;inner_k_count++){
            vload(Av1, &sa9(row_c,inner_k_count))
            vload(Av2, &sa9(row_c+4,inner_k_count))
            vload(Bv1, &sb9(col_c,inner_k_count))
            vload(Bv2, &sb9(col_c+4,inner_k_count))
            vscal(Cres[0], Av1, Bv1.x)
            vscal(Cres[1], Av2, Bv1.x)
            vscal(Cres[2], Av1, Bv1.y)
            vscal(Cres[3], Av2, Bv1.y)
            vscal(Cres[4], Av1, Bv1.z)
            vscal(Cres[5], Av2, Bv1.z)
            vscal(Cres[6], Av1, Bv1.w)
            vscal(Cres[7], Av2, Bv1.w)
            vscal(Cres[8], Av1, Bv2.x)
            vscal(Cres[9], Av2, Bv2.x)
            vscal(Cres[10], Av1, Bv2.y)
            vscal(Cres[11], Av2, Bv2.y)
            vscal(Cres[12], Av1, Bv2.z)
            vscal(Cres[13], Av2, Bv2.z)
            vscal(Cres[14], Av1, Bv2.w)
            vscal(Cres[15], Av2, Bv2.w)
        }
        __syncthreads();
    }
    vload(Cv[0], &C(row_c,col_c))
    vload(Cv[1], &C(row_c+4,col_c))
    vload(Cv[2], &C(row_c,col_c+1))
    vload(Cv[3], &C(row_c+4,col_c+1))
    vload(Cv[4], &C(row_c,col_c+2))
    vload(Cv[5], &C(row_c+4,col_c+2))
    vload(Cv[6], &C(row_c,col_c+3))
    vload(Cv[7], &C(row_c+4,col_c+3))
    vload(Cv[8], &C(row_c,col_c+4))
    vload(Cv[9], &C(row_c+4,col_c+4))
    vload(Cv[10], &C(row_c,col_c+5))
    vload(Cv[11], &C(row_c+4,col_c+5))
    vload(Cv[12], &C(row_c,col_c+6))
    vload(Cv[13], &C(row_c+4,col_c+6))
    vload(Cv[14], &C(row_c,col_c+7))
    vload(Cv[15], &C(row_c+4,col_c+7))
    
    simd_axpby(Cres[0],alpha,Cres[0],beta,Cv[0])
    simd_axpby(Cres[1],alpha,Cres[1],beta,Cv[1])
    simd_axpby(Cres[2],alpha,Cres[2],beta,Cv[2])
    simd_axpby(Cres[3],alpha,Cres[3],beta,Cv[3])

    simd_axpby(Cres[4],alpha,Cres[4],beta,Cv[4])
    simd_axpby(Cres[5],alpha,Cres[5],beta,Cv[5])
    simd_axpby(Cres[6],alpha,Cres[6],beta,Cv[6])
    simd_axpby(Cres[7],alpha,Cres[7],beta,Cv[7])

    simd_axpby(Cres[8],alpha,Cres[8],beta,Cv[8])
    simd_axpby(Cres[9],alpha,Cres[9],beta,Cv[9])
    simd_axpby(Cres[10],alpha,Cres[10],beta,Cv[10])
    simd_axpby(Cres[11],alpha,Cres[11],beta,Cv[11])

    simd_axpby(Cres[12],alpha,Cres[12],beta,Cv[12])
    simd_axpby(Cres[13],alpha,Cres[13],beta,Cv[13])
    simd_axpby(Cres[14],alpha,Cres[14],beta,Cv[14])
    simd_axpby(Cres[15],alpha,Cres[15],beta,Cv[15])

    vstore(&C(row_c,col_c), Cres[0])
    vstore(&C(row_c+4,col_c), Cres[1])
    vstore(&C(row_c,col_c+1), Cres[2])
    vstore(&C(row_c+4,col_c+1), Cres[3])
    vstore(&C(row_c,col_c+2), Cres[4])
    vstore(&C(row_c+4,col_c+2), Cres[5])
    vstore(&C(row_c,col_c+3), Cres[6])
    vstore(&C(row_c+4,col_c+3), Cres[7])
    vstore(&C(row_c,col_c+4), Cres[8])
    vstore(&C(row_c+4,col_c+4), Cres[9])
    vstore(&C(row_c,col_c+5), Cres[10])
    vstore(&C(row_c+4,col_c+5), Cres[11])
    vstore(&C(row_c,col_c+6), Cres[12])
    vstore(&C(row_c+4,col_c+6), Cres[13])
    vstore(&C(row_c,col_c+7), Cres[14])
    vstore(&C(row_c+4,col_c+7), Cres[15])
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
        // opt4_kernel<<<grid, block>>>(A, B, C, MM, KK, NN);
        // mysgemm_v8<<<grid, block>>>(MM, NN, KK, 1.0, A, B, 0.0, C);
        mysgemm_v9<<<grid, block>>>(MM, NN, KK, 1.0, A, B, 0.0, C);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);
    return time_elapsed;
}
