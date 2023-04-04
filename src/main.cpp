#include <string>
#include <iostream>
#include <chrono>
#include <tuple>
#include <cmath>
#include <algorithm>

#include "gemm.h"

using namespace std;

// C = A mul B
// A: M x K
// B: K x N
// C: M x N
int opt_level = -1;
int iter_num = 10;

void init(float *A, float *B, float *C_ref) {
    for(int ii = 0; ii < MM; ii++) {
        for (int jj = 0; jj < KK; jj++) {
            A[ii*KK+jj] = (MM-ii) * 1.f/MM + (KK-jj) * 1.f/KK;
        }
    }

    for(int ii = 0; ii < KK; ii++) {
        for (int jj = 0; jj < NN; jj++) {
            B[ii*NN+jj] = ii*1.f/KK + jj*1.f/NN;
        }
    }

    for (int ii = 0; ii < MM; ii ++) {
        for(int jj = 0; jj < NN; jj++) {
            C_ref[ii*NN+jj] = 0;
            for (int kk = 0; kk < KK; kk++) {
                C_ref[ii*NN+jj] += A[ii*KK+kk] * B[kk*NN+jj];
            }
        }
    }
}

// ret, i, j
tuple<bool, int, int> check_ret(float *C, float *C_ref) {
    for (int ii = 0; ii < MM; ii++) {
        for (int jj = 0; jj < NN; jj++) {
            if (abs(C[ii*NN+jj] - C_ref[ii*NN+jj]) / C[ii*NN+jj] > 1e-5) {
                return make_tuple(false, ii, jj);
            }
        }
    }
    return make_tuple(true, -1, -1);
}

void argparse(int argc, char *argv[]) {
    int cnt_argv = 1;
    while(argc > cnt_argv) {
        string cmd = argv[cnt_argv];
        if (cmd == "--shape") {
            std::cout << "[ERROR] No shape support" << std::endl;
            // M = stoi(argv[++cnt_argv]);
            // K = stoi(argv[++cnt_argv]);
            // N = stoi(argv[++cnt_argv]);
        } else if (cmd == "--opt") {
            opt_level = stoi(argv[++cnt_argv]);
        } else if (cmd == "--iter") {
            iter_num = stoi(argv[++cnt_argv]);
        } else {
            std::cout << "[ERROR] Incorrect arg" << std::endl;
        }
        cnt_argv++;
    }
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        argparse(argc, argv);
    }
    float* A = reinterpret_cast<float*>(malloc(MM * KK * sizeof(4)));
    float* B = reinterpret_cast<float*>(malloc(KK * NN * sizeof(4)));
    float* C = reinterpret_cast<float*>(malloc(MM * NN * sizeof(4)));
    float* C_ref = reinterpret_cast<float*>(malloc(MM * NN * sizeof(4)));

    init(A, B, C_ref);

    gemm(A, B, C, 1, opt_level);
    auto [err, err_ii, err_jj] = check_ret(C, C_ref);
    if (err != true) {
        std::cout << "[Error] C[ii][jj](" << C[err_ii*NN+err_jj] << ")" <<
         "!= C_ref[ii][jj](" << C_ref[err_ii*NN+err_jj] << "), while ii = " << 
         err_ii << ", jj = " << err_jj << std::endl;
         exit(1);
    } else {
        std::cout << "[INFO] PASS" << std::endl;
    }
    std::cout << "[INFO] iter : " << iter_num << std::endl;
    auto time_cost_ms = gemm(A, B, C, iter_num, opt_level);
    std::cout << "[INFO] time : " << time_cost_ms << "ms" << std::endl;
    std::cout << "[INFO] GFLOPs : " << MM*NN*KK / 1e9 / time_cost_ms * 2000 * iter_num
              << " GFLOPS" << std::endl;

    std::cout << "[INFO] peak : " <<  MM*NN*KK / 1e9 / time_cost_ms / 10.07 * 200 *iter_num << "%" << std::endl;
    
}