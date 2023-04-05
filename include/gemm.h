#ifndef __INCLUDE_GEMM_H__
#define __INCLUDE_GEMM_H__

#define MM 4096
#define NN 4096
#define KK 4096
float gemm(float *, float *, float *, int iter, int opt);
float baseline(float *, float *, float *, int iter);
float opt0(float *, float *, float *, int iter);
float opt05(float *, float *, float *, int iter);
float opt1(float *, float *, float *, int iter);
float opt2(float *, float *, float *, int iter);

#endif