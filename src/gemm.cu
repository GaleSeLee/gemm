#include <cuda_runtime.h>
#include "gemm.h"

float gemm(float *A, float *B, float *C, int iter, int opt) {
    float *dev_A, *dev_B, *dev_C;
    cudaMalloc((void**)&dev_A, MM*KK*sizeof(float));
    cudaMalloc((void**)&dev_B, KK*NN*sizeof(float));
    cudaMalloc((void**)&dev_C, MM*NN*sizeof(float));
    cudaMemcpy(dev_A, A, sizeof(float)*MM*KK, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, sizeof(float)*KK*NN, cudaMemcpyHostToDevice);
    float time_elapsed = 0.0;
    switch(opt) {
        case 0:
            break;
        default:
            time_elapsed = baseline(dev_A, dev_B, dev_C, iter);
            break;
    }
    cudaMemcpy(C, dev_C, sizeof(float) * MM * NN, cudaMemcpyDeviceToHost);

    return time_elapsed;
}