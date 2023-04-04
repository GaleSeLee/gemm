#ifndef __INCLUDE_GEMM_H__
#define __INCLUDE_GEMM_H__

#define MM 1024
#define NN 1024
#define KK 1024
float gemm(float *, float *, float *, int iter, int opt);
float baseline(float *, float *, float *, int iter);

#endif