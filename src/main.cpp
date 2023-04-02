#include <string>
#include <iostream>
#include <chrono>
#include <tuple>

#include "gemm.h"

using namespace std;

// C = A mul B
// A: M x K
// B: K x N
// C: M x N
int M = 256, K = 256, N = 256;

void init(int *A, int *B, int *C_ref) {
    for(int ii = 0; ii < M; ii++) {
        for (int jj = 0; jj < K; jj++) {
            A[ii*K+jj] = ii * K + jj;
        }
    }

    for(int ii = 0; ii < K; ii++) {
        for (int jj = 0; jj < N; jj++) {
            B[ii*N+jj] = K * N - ii * N - jj;
        }
    }

    for (int ii = 0; ii < M; ii ++) {
        for(int jj = 0; jj < N; jj++) {
            C_ref[ii*N+jj] = 0;
            for (int kk = 0; kk < K; kk++) {
                C_ref[ii*N+jj] += A[ii*K+kk] * B[kk*N+jj];
            }
        }
    }
}

// ret, i, j
tuple<bool, int, int> check_ret(int *C, int *C_ref) {
    for (int ii = 0; ii < M; ii++) {
        for (int jj = 0; jj < N; jj++) {
            if (C[ii*N+jj] != C_ref[ii*N+jj]) {
                return make_tuple(false, ii, jj);
            }
        }
    }
    return make_tuple(true, -1, -1);
}

int main(int argc, char *argv[]) {
    if (argc == 4) {
        M = stoi(argv[1]);
        K = stoi(argv[2]);
        N = stoi(argv[3]);
    }
    int* A = reinterpret_cast<int*>(malloc(M * K * sizeof(4)));
    int* B = reinterpret_cast<int*>(malloc(K * N * sizeof(4)));
    int* C = reinterpret_cast<int*>(malloc(M * N * sizeof(4)));
    int* C_ref = reinterpret_cast<int*>(malloc(M * N * sizeof(4)));

    init(A, B, C_ref);

    gemm_baseline(A, B, C);
    auto [err, err_ii, err_jj] = check_ret(C, C_ref);
    if (err != true) {
        std::cout << "[Error] C[ii][jj](" << C[err_ii*N+err_jj] << ")" <<
         "!= C_ref[ii][jj](" << C_ref[err_ii*N+err_jj] << "), while ii = " << 
         err_ii << "jj = " << err_jj << std::endl;
    } else {
        std::cout << "[INFO] PASS" << std::endl;
    }

    auto start = std::chrono::system_clock::now();

    for (int ii = 0; ii < 10; ii++) {
        gemm_baseline(A, B, C);
    }
    
    auto end = std::chrono::system_clock::now();
    std::cout << "time : " \
         << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() / 1000.0
         << "ms" << std::endl;
}