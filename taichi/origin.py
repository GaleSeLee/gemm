from argparse import MetavarTypeHelpFormatter
import taichi as ti
import numpy as np
# from core.taichi import TaichiBenchmark
import argparse

data_type = ti.float32
arr_type = ti.types.ndarray(dtype=data_type, ndim=2)


def ref_sgemm(M:int, N:int, K:int, 
             alpha:float, A,
             beta:float,  B,
             C):
    # print("A: ")
    # print(A)
    # print("B: ")
    # print(B)
    # print("res: ")
    # print("{} {}".format(alpha, beta))
    # print(np.dot(A, B))
    C = alpha * np.dot(A, B) + beta * C
    # print("C")
    # print(C)
    return C

def verify_matrix(a, b):
    if a.shape != b.shape:
       return False
    is_all_close = np.allclose(a, b, rtol=1e-5, atol=1e-08)
    if is_all_close:
       return True
    else:
      #  return False
      print("not compute right!!")
      # not_close_indices = np.where(np.logical_not(np.isclose(a, b, rtol=1e-05, atol=1e-08)))
      # count  = 0
      # for row, col in zip(not_close_indices[0], not_close_indices[1]):
      #    count+= 1
      #    print("({}, {}: {} {})".format(row, col, a[row, col], b[row, col]), end=" ")
      #    if count == 4:
      #       print()
      #       count = 0
      exit(-1)

MS = KS = NS = 32
block_size = MS * NS

@ti.kernel
def sgemm_v1(M:int, N:int, K:int, 
             alpha:float, A: arr_type,
             beta:float,  B:arr_type,
             C: arr_type):
    
    ti.loop_config(block_dim=256)
    for i, j in ti.ndrange(M, N):
        # for j in range(N):
        tmp = 0.0
        for k_count in range(K):
          tmp += A[i, k_count] * B[k_count, j]

        C[i,j] = alpha * tmp + beta * C[i, j]

# block_size = 256 

@ti.kernel
def sgemm_v2(M:int, N:int, K:int, 
             alpha:float, A: arr_type,
             beta:float,  B:arr_type,
             C: arr_type):
    # print("in sgemm_v2")
    row_grid = M // MS
    col_grid = N // NS
    total_block = row_grid * col_grid
    # print("total block we need: ", total_block)
    ti.loop_config(block_dim=block_size)
    for i, j in ti.ndrange(M, N):
    
        g_tid = ti.simt.block.global_thread_idx()
        tid = g_tid % block_size
        block_id = (g_tid // block_size) % (total_block)
        block_idx = block_id % row_grid
        block_idy = block_id // col_grid 
        row_ptr = block_idx * MS
        col_ptr = block_idy * NS
        col = tid % NS
        row = tid // MS
    
        sa = ti.simt.block.SharedArray((MS,KS), ti.f32)
        sb = ti.simt.block.SharedArray((KS,NS), ti.f32)

        tmp = 0.
        for k_count in range(K // KS):
          sa[row, col] = A[row_ptr + row, col + k_count * KS]
          sb[row, col] = B[row + k_count * KS,  col_ptr + col]
          ti.simt.block.sync()
          for innner_k_count in range(KS):
            tmp += sa[row, innner_k_count] * sb[innner_k_count, col]
          ti.simt.block.sync()
        C[row_ptr + row,col_ptr + col] = alpha * tmp + beta * C[row_ptr + row, col_ptr + col]


# every thread compute 4x1 micro kernel C(1x4) = A(1xKS) * B (KS x 4)
@ti.kernel
def sgemm_v3(M:int, N:int, K:int, 
             alpha:float, A: arr_type,
             beta:float,  B:arr_type,
             C: arr_type):
    # print("in sgemm_v2")
   
    row_grid = M // MS
    col_grid = N // NS
    total_block = (M * N) // (MS * NS)
    ti.loop_config(block_dim=block_size)
    for i, j in ti.ndrange(M, N):
    
        g_tid = ti.simt.block.global_thread_idx()
        tid = g_tid % (block_size)
        block_id = (g_tid // block_size) % total_block
        block_idx = block_id % row_grid
        block_idy = block_id // col_grid
       
        row_ptr = block_idx * MS
        col_ptr = block_idy * NS
        col = (tid % (NS // 4)) <<2 #0

        row = tid // (block_size // NS)
    
        sa = ti.simt.block.SharedArray((MS,KS), ti.f32)
        sb = ti.simt.block.SharedArray((KS,NS), ti.f32)

        Cres_0 =  Cres_1 = Cres_2 = Cres_3 = 0.
        for k_count in range(K // KS):
          sa[row, col]   = A[row_ptr + row,   col   + k_count * KS]
          sa[row, col+1] = A[row_ptr + row,   col+1 + k_count * KS]
          sa[row, col+2] = A[row_ptr + row,   col+2 + k_count * KS]
          sa[row, col+3] = A[row_ptr + row,   col+3 + k_count * KS]
          sb[row, col]   = B[row + k_count * KS,  col_ptr + col]
          sb[row, col+1] = B[row + k_count * KS,  col_ptr + col+1]
          sb[row, col+2] = B[row + k_count * KS,  col_ptr + col+2]
          sb[row, col+3] = B[row + k_count * KS,  col_ptr + col+3]
          ti.simt.block.sync()
          for innner_k_count in range(KS):
            Cres_0 += sa[row, innner_k_count] * sb[innner_k_count, col]
            Cres_1 += sa[row, innner_k_count] * sb[innner_k_count, col+1]
            Cres_2 += sa[row, innner_k_count] * sb[innner_k_count, col+2]
            Cres_3 += sa[row, innner_k_count] * sb[innner_k_count, col+3]
          ti.simt.block.sync()
        C[row_ptr + row,col_ptr + col]   = alpha * Cres_0 + beta * C[row_ptr + row, col_ptr + col]
        C[row_ptr + row,col_ptr + col+1] = alpha * Cres_1 + beta * C[row_ptr + row, col_ptr + col+1]
        C[row_ptr + row,col_ptr + col+2] = alpha * Cres_2 + beta * C[row_ptr + row, col_ptr + col+2]
        C[row_ptr + row,col_ptr + col+3] = alpha * Cres_3 + beta * C[row_ptr + row, col_ptr + col+3]

# every thread compute 4x4 micro kernel C(4x4) = A(4xKS) * B (KS x 4)

# KS = 16
# we have a block of 16 * 16 thread each thread compute 4* 4 kernel
# each block 64 x 64, we have block (M // 64) * (N // 64)
# each thread compute 4 x 4 kernel, 
# each block: A [64 x 16] B[16 x 64] C [64 x 64]

@ti.kernel
def sgemm_v4(M:int, N:int, K:int, 
             alpha:float, A: arr_type,
             beta:float,  B:arr_type,
             C: arr_type):
    # print("in sgemm_v2")
   
    row_grid = M // MS
    col_grid = N // NS
    total_block = row_grid * col_grid
    ti.loop_config(block_dim=block_size)
    for i, j in ti.ndrange(M, N):
    
        g_tid = ti.simt.block.global_thread_idx()
        tid = g_tid % (block_size)
        block_id = (g_tid // block_size) % total_block
        block_idx = block_id % row_grid
        block_idy = block_id // col_grid
       
        row_ptr = block_idx * MS
        col_ptr = block_idy * NS
        row_a = tid >> 2#tid // 4
        col_a = (tid & 3) << 2#(tid % 4) << 2 
        row_b = tid >> 4 #tid // 16
        col_b = (tid & 15) << 2
        row_c = (tid >> 4) << 2
        col_c = (tid & 15) << 2
    
        # sa = ti.simt.block.SharedArray((MS,KS), ti.f32)
        sa = ti.simt.block.SharedArray((KS,MS), ti.f32)
        sb = ti.simt.block.SharedArray((KS,NS), ti.f32)

        Cres_0     = ti.math.vec4(0., 0., 0., 0.)
        Cres_1     = ti.math.vec4(0., 0., 0., 0.)
        Cres_2     = ti.math.vec4(0., 0., 0., 0.)
        Cres_3     = ti.math.vec4(0., 0., 0., 0.)
        for k_count in range(K // KS):
          # sa[row_a, col_a]   = A[row_ptr + row_a,   col_a   + k_count * KS]
          # sa[row_a, col_a+1] = A[row_ptr + row_a,   col_a+1 + k_count * KS]
          # sa[row_a, col_a+2] = A[row_ptr + row_a,   col_a+2 + k_count * KS]
          # sa[row_a, col_a+3] = A[row_ptr + row_a,   col_a+3 + k_count * KS]
          sa[col_a,   row_a] = A[row_ptr + row_a,   col_a   + k_count * KS]
          sa[col_a+1, row_a] = A[row_ptr + row_a,   col_a+1 + k_count * KS]
          sa[col_a+2, row_a] = A[row_ptr + row_a,   col_a+2 + k_count * KS]
          sa[col_a+3, row_a] = A[row_ptr + row_a,   col_a+3 + k_count * KS]
          sb[row_b, col_b]   = B[row_b + k_count * KS,  col_ptr + col_b]
          sb[row_b, col_b+1] = B[row_b + k_count * KS,  col_ptr + col_b+1]
          sb[row_b, col_b+2] = B[row_b + k_count * KS,  col_ptr + col_b+2]
          sb[row_b, col_b+3] = B[row_b + k_count * KS,  col_ptr + col_b+3]
          ti.simt.block.sync()
          for innner_k_count in ti.static(range(KS)):
            sa_0 = sa[innner_k_count, row_c]
            sa_1 = sa[innner_k_count, row_c+1]
            sa_2 = sa[innner_k_count, row_c+2]
            sa_3 = sa[innner_k_count, row_c+3]

            sb_0 = sb[innner_k_count, col_c]
            sb_1 = sb[innner_k_count, col_c+1]
            sb_2 = sb[innner_k_count, col_c+2]
            sb_3 = sb[innner_k_count, col_c+3]

            Cres_0[0] += sa_0 * sb_0
            Cres_0[1] += sa_0 * sb_1
            Cres_0[2] += sa_0 * sb_2
            Cres_0[3] += sa_0 * sb_3

            Cres_1[0] += sa_1 * sb_0
            Cres_1[1] += sa_1 * sb_1
            Cres_1[2] += sa_1 * sb_2
            Cres_1[3] += sa_1 * sb_3

            Cres_2[0] += sa_2 * sb_0
            Cres_2[1] += sa_2 * sb_1
            Cres_2[2] += sa_2 * sb_2
            Cres_2[3] += sa_2 * sb_3

            Cres_3[0] += sa_3 * sb_0
            Cres_3[1] += sa_3 * sb_1
            Cres_3[2] += sa_3 * sb_2
            Cres_3[3] += sa_3 * sb_3
            # Cres_1 += sa[row, innner_k_count] * sb[innner_k_count, col+1]
            # Cres_2 += sa[row, innner_k_count] * sb[innner_k_count, col+2]
            # Cres_3 += sa[row, innner_k_count] * sb[innner_k_count, col+3]
          ti.simt.block.sync()
        C[row_ptr + row_c,col_ptr + col_c]     = alpha * Cres_0[0] + beta * C[row_ptr + row_c, col_ptr + col_c]
        C[row_ptr + row_c,col_ptr + col_c+1]   = alpha * Cres_0[1] + beta * C[row_ptr + row_c, col_ptr + col_c+1]
        C[row_ptr + row_c,col_ptr + col_c+2]   = alpha * Cres_0[2] + beta * C[row_ptr + row_c, col_ptr + col_c+2]
        C[row_ptr + row_c,col_ptr + col_c+3]   = alpha * Cres_0[3] + beta * C[row_ptr + row_c, col_ptr + col_c+3]
        

        C[row_ptr + row_c+1,col_ptr + col_c]     = alpha * Cres_1[0] + beta * C[row_ptr + row_c+1, col_ptr + col_c]
        C[row_ptr + row_c+1,col_ptr + col_c+1]   = alpha * Cres_1[1] + beta * C[row_ptr + row_c+1, col_ptr + col_c+1]
        C[row_ptr + row_c+1,col_ptr + col_c+2]   = alpha * Cres_1[2] + beta * C[row_ptr + row_c+1, col_ptr + col_c+2]
        C[row_ptr + row_c+1,col_ptr + col_c+3]   = alpha * Cres_1[3] + beta * C[row_ptr + row_c+1, col_ptr + col_c+3]

        C[row_ptr + row_c+2,col_ptr + col_c]     = alpha * Cres_2[0] + beta * C[row_ptr + row_c+2, col_ptr + col_c]
        C[row_ptr + row_c+2,col_ptr + col_c+1]   = alpha * Cres_2[1] + beta * C[row_ptr + row_c+2, col_ptr + col_c+1]
        C[row_ptr + row_c+2,col_ptr + col_c+2]   = alpha * Cres_2[2] + beta * C[row_ptr + row_c+2, col_ptr + col_c+2]
        C[row_ptr + row_c+2,col_ptr + col_c+3]   = alpha * Cres_2[3] + beta * C[row_ptr + row_c+2, col_ptr + col_c+3]

        C[row_ptr + row_c+3,col_ptr + col_c]     = alpha * Cres_3[0] + beta * C[row_ptr + row_c+3, col_ptr + col_c]
        C[row_ptr + row_c+3,col_ptr + col_c+1]   = alpha * Cres_3[1] + beta * C[row_ptr + row_c+3, col_ptr + col_c+1]
        C[row_ptr + row_c+3,col_ptr + col_c+2]   = alpha * Cres_3[2] + beta * C[row_ptr + row_c+3, col_ptr + col_c+2]
        C[row_ptr + row_c+3,col_ptr + col_c+3]   = alpha * Cres_3[3] + beta * C[row_ptr + row_c+3, col_ptr + col_c+3]


class Gemm:
   
   name = 'gemm'
   size = [(i+1) << 8 for i in range(24)]
   gemm_kernel_switcher = {
           1: sgemm_v1,
           2: sgemm_v2,
           3: sgemm_v3,
           4: sgemm_v4
        }
   

   def get_gemm(self, kernel_num):
      global block_size, MS, NS, KS   
      if kernel_num == 1 or kernel_num == 2:
         MS = NS = KS = 32
         block_size = MS * NS
      elif kernel_num == 3:
         MS =NS = KS = 32
         block_size = 256
      elif kernel_num == 4:
         MS = NS = 64
         KS = 16
         block_size = 256
      return self.gemm_kernel_switcher.get(kernel_num)
   
   def create_A_B_C(self, m, n, k):
      np_A = np.random.rand(m, k).astype(np.float32)
      np_A.fill(1.0)
      np_B = np.random.rand(k, n).astype(np.float32)
      np_B.fill(1.0)
      np_C = np.random.rand(m, n).astype(np.float32)
      np_C.fill(0.0)
      A = ti.ndarray(dtype=data_type,shape=(m,k))
      # A.fill(1)
      B = ti.ndarray(dtype=data_type,shape=(k,n))
      # B.fill(2)
      C = ti.ndarray(dtype=data_type,shape=(m,n))
      A.from_numpy(np_A)
      B.from_numpy(np_B)
      C.from_numpy(np_C)
      return A, B, C, np_A, np_B, np_C
   
   def call_gemm(self, size: int, func):
      m = n = k = size
      print("M=N=K: ", size)
      alpha = 1.0
      beta  = 0.0
      A, B, C, np_A, np_B, np_C = self.create_A_B_C(m, n, k)
      np_C = ref_sgemm(m,n,k, alpha, np_A, beta, np_B, np_C)

      # func= self.gemm_kernel_switcher.get(kernel_number)

      func(m,n,k,
        alpha, A, beta, B, C)
      # print("np_C")
      # print(np_C)
      # print("C")
      # print(C.to_numpy())

      if not verify_matrix(np_C, C.to_numpy()):
         print("not compute right!!!")
         exit(-1)
      repeats = 100
      for _ in range(repeats):
          func(m,n,k, alpha, A, beta, B, C)
      query_result = ti.profiler.query_kernel_profiler_info(func.__name__)
      avg_time = query_result.avg
      flops = 2.*1e-6 * m * n * k / avg_time
      print("kernel elapsed time(avg_in_ms) {} gflops {}".format(query_result.avg, flops))

      print(f"peak {flops/6451*100}%")

   def init(self, size: int, kernel_number= 1):
      gemm_func = self.get_gemm(kernel_number)
      if size > 0:
        self.call_gemm(size, gemm_func)
      else:
         for sz in self.size:
            self.call_gemm(sz, gemm_func)




# gemm_block = Gemm()
# gemm_block.init(1536, 1)       

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='parse gemm matrix size and which kernel we use')
  parser.add_argument('--size', type=int, help='matrix M, K, N size')
  parser.add_argument('--kernel', type=int, help='which gemm kernel we use')
  parser.add_argument('--help_info', type=str,  help='show some help info')

  args = parser.parse_args()
  if not any(vars(args).values()):
    print("you should set --size(-1, 128, 256, 512) to set matrix size and -- kernel(1, 2, 3) to set which kernel we use")
    exit(-1)

  
  ti.init(arch = ti.cuda,
        kernel_profiler=True,
        print_ir=False)
  gemm_block = Gemm()
  gemm_block.init(args.size, args.kernel)

# repeats = 100
# print("M=N={} ".format(m) )

# sgemm_v2(m,n,k,
#          alpha, A, beta, B,
#          C)

# for _ in range(repeats):
#     sgemm_v1(m,n,k,
#          alpha, A,
#          beta, B,
#          C)
# query_result = ti.profiler.query_kernel_profiler_info(sgemm_v1.__name__)
# avg_time = query_result.avg
# flops = 2.*1e-6 * MetavarTypeHelpFormatter * n * k / avg_time
# print("kernel elapsed time(avg_in_ms) {} gflops {}".format(query_result.avg, flops))
# print("kernel elapsed time(avg_in_ms) =",query_result.avg)
