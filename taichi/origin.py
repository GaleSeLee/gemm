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
    for i in range(128):
      for j in range(4):
        if a[i][j] != b[i][j]:
          print(f"{i}, {j}: {a[i][j]}, {b[i][j]}")
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


block_dim_x = 16
block_dim_y = 16
element_per_thread_x = 8
element_per_thread_y = 8
block_size = 16 * 16
@ti.kernel
def sgemm_128_128_kernel(M:int, N:int, K:int, 
             alpha:float, A: arr_type,
             beta:float,  B:arr_type,
             C: arr_type):
  # 128 times 128 currently
  row_grid = M // (block_dim_x * 8)
  col_grid = N // (block_dim_y * 8)
  total_block = row_grid * col_grid
  ti.loop_config(block_dim=block_size)
  for i, j in ti.ndrange(M // 8, N // 8):
    g_tid = ti.simt.block.global_thread_idx()
    tid = g_tid % (block_size)
    warp_id = tid >> 5
    warp_row = warp_id >> 1
    warp_col = warp_id & 1
    lane_id = tid & 31

    block_id = g_tid // block_size
    block_idx = block_id % col_grid
    block_idy = block_id // col_grid

    global_A_row = tid >> 1 + block_idx * 128
    global_A_col = (tid & 1) << 2
    global_B_row = warp_id
    global_B_col = block_idy * 128 + lane_id * 4

    global_C_row_base = block_idx * 128 + warp_row * 32 + lane_id // 8 * 4
    global_C_col_base = block_idy * 128 + warp_col * 64 + lane_id % 8 * 4
    global_C_row_delta = global_C_row_base + 16
    global_C_col_delta = global_C_col_base + 32

    block_A_row_base = global_C_row_base - block_idx * 128
    block_A_row_delta = global_C_row_delta - block_idx * 128
    block_B_col_base = global_C_col_base - block_idy * 128
    block_B_col_delta = global_C_col_delta - block_idy * 128

    block_A = ti.simt.block.SharedArray((1024, ), ti.f32)
    block_B = ti.simt.block.SharedArray((1024, ), ti.f32)
    a_frag_0 = ti.math.vec4(0., 0., 0., 0.)
    a_frag_1 = ti.math.vec4(0., 0., 0., 0.)
    b_frag_0 = ti.math.vec4(0., 0., 0., 0.)
    b_frag_1 = ti.math.vec4(0., 0., 0., 0.)

    c_frag_0_0 = ti.math.vec4(0., 0., 0., 0.)
    c_frag_0_1 = ti.math.vec4(0., 0., 0., 0.)
    c_frag_0_2 = ti.math.vec4(0., 0., 0., 0.)
    c_frag_0_3 = ti.math.vec4(0., 0., 0., 0.)

    c_frag_1_0 = ti.math.vec4(0., 0., 0., 0.)
    c_frag_1_1 = ti.math.vec4(0., 0., 0., 0.)
    c_frag_1_2 = ti.math.vec4(0., 0., 0., 0.)
    c_frag_1_3 = ti.math.vec4(0., 0., 0., 0.)

    c_frag_2_0 = ti.math.vec4(0., 0., 0., 0.)
    c_frag_2_1 = ti.math.vec4(0., 0., 0., 0.)
    c_frag_2_2 = ti.math.vec4(0., 0., 0., 0.)
    c_frag_2_3 = ti.math.vec4(0., 0., 0., 0.)

    c_frag_3_0 = ti.math.vec4(0., 0., 0., 0.)
    c_frag_3_1 = ti.math.vec4(0., 0., 0., 0.)
    c_frag_3_2 = ti.math.vec4(0., 0., 0., 0.)
    c_frag_3_3 = ti.math.vec4(0., 0., 0., 0.)

    for kk in range(K // 8):
      ii = kk * 8
      block_A[tid*4] = A[global_A_row, global_A_col + ii]
      block_A[tid*4+1] = A[global_A_row, global_A_col + ii + 1]
      block_A[tid*4+2] = A[global_A_row, global_A_col + ii + 2]
      block_A[tid*4+3] = A[global_A_row, global_A_col + ii + 3]

      block_B[tid*4] = B[global_B_row+ii, global_B_col]
      block_B[tid*4+1] = B[global_B_row+ii, global_B_col+1]
      block_B[tid*4+2] = B[global_B_row+ii, global_B_col+2]
      block_B[tid*4+3] = B[global_B_row+ii, global_B_col+3]
      ti.simt.block.sync()

      #if tid*4 + 3 == 903:
      #  print("xidx = ", global_B_row+ii+3, global_B_col, ii, global_B_row)

      block_A_base_id = block_A_row_base * 8
      block_A_delta_id = block_A_row_delta * 8

      # how to unroll
      for jj in ti.static(range(8)):
        a_frag_0[0] = block_A[block_A_base_id+jj]
        a_frag_0[1] = block_A[block_A_base_id+jj+8]
        a_frag_0[2] = block_A[block_A_base_id+jj+16]
        a_frag_0[3] = block_A[block_A_base_id+jj+24]
        a_frag_1[0] = block_A[block_A_delta_id+jj]
        a_frag_1[1] = block_A[block_A_delta_id+jj+8]
        a_frag_1[2] = block_A[block_A_delta_id+jj+16]
        a_frag_1[3] = block_A[block_A_delta_id+jj+24]

        b_frag_0[0] = block_B[jj*128 + block_B_col_base]
        b_frag_0[1] = block_B[jj*128 + block_B_col_base+1]
        b_frag_0[2] = block_B[jj*128 + block_B_col_base+2]
        b_frag_0[3] = block_B[jj*128 + block_B_col_base+3]
        b_frag_1[0] = block_B[jj*128 + block_B_col_delta]
        b_frag_1[1] = block_B[jj*128 + block_B_col_delta+1]
        b_frag_1[2] = block_B[jj*128 + block_B_col_delta+2]
        b_frag_1[3] = block_B[jj*128 + block_B_col_delta+3]
      

        c_frag_0_0[0] += a_frag_0[0] * b_frag_0[0]
        c_frag_0_0[1] += a_frag_0[0] * b_frag_0[1]
        c_frag_0_0[2] += a_frag_0[0] * b_frag_0[2]
        c_frag_0_0[3] += a_frag_0[0] * b_frag_0[3]
        c_frag_0_1[0] += a_frag_0[1] * b_frag_0[0]
        c_frag_0_1[1] += a_frag_0[1] * b_frag_0[1]
        c_frag_0_1[2] += a_frag_0[1] * b_frag_0[2]
        c_frag_0_1[3] += a_frag_0[1] * b_frag_0[3]
        c_frag_0_2[0] += a_frag_0[2] * b_frag_0[0]
        c_frag_0_2[1] += a_frag_0[2] * b_frag_0[1]
        c_frag_0_2[2] += a_frag_0[2] * b_frag_0[2]
        c_frag_0_2[3] += a_frag_0[2] * b_frag_0[3]
        c_frag_0_3[0] += a_frag_0[3] * b_frag_0[0]
        c_frag_0_3[1] += a_frag_0[3] * b_frag_0[1]
        c_frag_0_3[2] += a_frag_0[3] * b_frag_0[2]
        c_frag_0_3[3] += a_frag_0[3] * b_frag_0[3]
        c_frag_1_0[0] += a_frag_0[0] * b_frag_1[0]
        c_frag_1_0[1] += a_frag_0[0] * b_frag_1[1]
        c_frag_1_0[2] += a_frag_0[0] * b_frag_1[2]
        c_frag_1_0[3] += a_frag_0[0] * b_frag_1[3]
        c_frag_1_1[0] += a_frag_0[1] * b_frag_1[0]
        c_frag_1_1[1] += a_frag_0[1] * b_frag_1[1]
        c_frag_1_1[2] += a_frag_0[1] * b_frag_1[2]
        c_frag_1_1[3] += a_frag_0[1] * b_frag_1[3]
        c_frag_1_2[0] += a_frag_0[2] * b_frag_1[0]
        c_frag_1_2[1] += a_frag_0[2] * b_frag_1[1]
        c_frag_1_2[2] += a_frag_0[2] * b_frag_1[2]
        c_frag_1_2[3] += a_frag_0[2] * b_frag_1[3]
        c_frag_1_3[0] += a_frag_0[3] * b_frag_1[0]
        c_frag_1_3[1] += a_frag_0[3] * b_frag_1[1]
        c_frag_1_3[2] += a_frag_0[3] * b_frag_1[2]
        c_frag_1_3[3] += a_frag_0[3] * b_frag_1[3]
        c_frag_2_0[0] += a_frag_1[0] * b_frag_0[0]
        c_frag_2_0[1] += a_frag_1[0] * b_frag_0[1]
        c_frag_2_0[2] += a_frag_1[0] * b_frag_0[2]
        c_frag_2_0[3] += a_frag_1[0] * b_frag_0[3]
        c_frag_2_1[0] += a_frag_1[1] * b_frag_0[0]
        c_frag_2_1[1] += a_frag_1[1] * b_frag_0[1]
        c_frag_2_1[2] += a_frag_1[1] * b_frag_0[2]
        c_frag_2_1[3] += a_frag_1[1] * b_frag_0[3]
        c_frag_2_2[0] += a_frag_1[2] * b_frag_0[0]
        c_frag_2_2[1] += a_frag_1[2] * b_frag_0[1]
        c_frag_2_2[2] += a_frag_1[2] * b_frag_0[2]
        c_frag_2_2[3] += a_frag_1[2] * b_frag_0[3]
        c_frag_2_3[0] += a_frag_1[3] * b_frag_0[0]
        c_frag_2_3[1] += a_frag_1[3] * b_frag_0[1]
        c_frag_2_3[2] += a_frag_1[3] * b_frag_0[2]
        c_frag_2_3[3] += a_frag_1[3] * b_frag_0[3]
        c_frag_3_0[0] += a_frag_1[0] * b_frag_1[0]
        c_frag_3_0[1] += a_frag_1[0] * b_frag_1[1]
        c_frag_3_0[2] += a_frag_1[0] * b_frag_1[2]
        c_frag_3_0[3] += a_frag_1[0] * b_frag_1[3]
        c_frag_3_1[0] += a_frag_1[1] * b_frag_1[0]
        c_frag_3_1[1] += a_frag_1[1] * b_frag_1[1]
        c_frag_3_1[2] += a_frag_1[1] * b_frag_1[2]
        c_frag_3_1[3] += a_frag_1[1] * b_frag_1[3]
        c_frag_3_2[0] += a_frag_1[2] * b_frag_1[0]
        c_frag_3_2[1] += a_frag_1[2] * b_frag_1[1]
        c_frag_3_2[2] += a_frag_1[2] * b_frag_1[2]
        c_frag_3_2[3] += a_frag_1[2] * b_frag_1[3]
        c_frag_3_3[0] += a_frag_1[3] * b_frag_1[0]
        c_frag_3_3[1] += a_frag_1[3] * b_frag_1[1]
        c_frag_3_3[2] += a_frag_1[3] * b_frag_1[2]
        c_frag_3_3[3] += a_frag_1[3] * b_frag_1[3]
        

      ti.simt.block.sync()
    C[global_C_row_base+0, global_C_col_base+0] = c_frag_0_0[0]
    C[global_C_row_base+0, global_C_col_base+1] = c_frag_0_0[1]
    C[global_C_row_base+0, global_C_col_base+2] = c_frag_0_0[2]
    C[global_C_row_base+0, global_C_col_base+3] = c_frag_0_0[3]
    C[global_C_row_base+1, global_C_col_base+0] = c_frag_0_1[0]
    C[global_C_row_base+1, global_C_col_base+1] = c_frag_0_1[1]
    C[global_C_row_base+1, global_C_col_base+2] = c_frag_0_1[2]
    C[global_C_row_base+1, global_C_col_base+3] = c_frag_0_1[3]
    C[global_C_row_base+2, global_C_col_base+0] = c_frag_0_2[0]
    C[global_C_row_base+2, global_C_col_base+1] = c_frag_0_2[1]
    C[global_C_row_base+2, global_C_col_base+2] = c_frag_0_2[2]
    C[global_C_row_base+2, global_C_col_base+3] = c_frag_0_2[3]
    C[global_C_row_base+3, global_C_col_base+0] = c_frag_0_3[0]
    C[global_C_row_base+3, global_C_col_base+1] = c_frag_0_3[1]
    C[global_C_row_base+3, global_C_col_base+2] = c_frag_0_3[2]
    C[global_C_row_base+3, global_C_col_base+3] = c_frag_0_3[3]
    C[global_C_row_base+0, global_C_col_delta+0] = c_frag_1_0[0]
    C[global_C_row_base+0, global_C_col_delta+1] = c_frag_1_0[1]
    C[global_C_row_base+0, global_C_col_delta+2] = c_frag_1_0[2]
    C[global_C_row_base+0, global_C_col_delta+3] = c_frag_1_0[3]
    C[global_C_row_base+1, global_C_col_delta+0] = c_frag_1_1[0]
    C[global_C_row_base+1, global_C_col_delta+1] = c_frag_1_1[1]
    C[global_C_row_base+1, global_C_col_delta+2] = c_frag_1_1[2]
    C[global_C_row_base+1, global_C_col_delta+3] = c_frag_1_1[3]
    C[global_C_row_base+2, global_C_col_delta+0] = c_frag_1_2[0]
    C[global_C_row_base+2, global_C_col_delta+1] = c_frag_1_2[1]
    C[global_C_row_base+2, global_C_col_delta+2] = c_frag_1_2[2]
    C[global_C_row_base+2, global_C_col_delta+3] = c_frag_1_2[3]
    C[global_C_row_base+3, global_C_col_delta+0] = c_frag_1_3[0]
    C[global_C_row_base+3, global_C_col_delta+1] = c_frag_1_3[1]
    C[global_C_row_base+3, global_C_col_delta+2] = c_frag_1_3[2]
    C[global_C_row_base+3, global_C_col_delta+3] = c_frag_1_3[3]
    C[global_C_row_delta+0, global_C_col_base+0] = c_frag_2_0[0]
    C[global_C_row_delta+0, global_C_col_base+1] = c_frag_2_0[1]
    C[global_C_row_delta+0, global_C_col_base+2] = c_frag_2_0[2]
    C[global_C_row_delta+0, global_C_col_base+3] = c_frag_2_0[3]
    C[global_C_row_delta+1, global_C_col_base+0] = c_frag_2_1[0]
    C[global_C_row_delta+1, global_C_col_base+1] = c_frag_2_1[1]
    C[global_C_row_delta+1, global_C_col_base+2] = c_frag_2_1[2]
    C[global_C_row_delta+1, global_C_col_base+3] = c_frag_2_1[3]
    C[global_C_row_delta+2, global_C_col_base+0] = c_frag_2_2[0]
    C[global_C_row_delta+2, global_C_col_base+1] = c_frag_2_2[1]
    C[global_C_row_delta+2, global_C_col_base+2] = c_frag_2_2[2]
    C[global_C_row_delta+2, global_C_col_base+3] = c_frag_2_2[3]
    C[global_C_row_delta+3, global_C_col_base+0] = c_frag_2_3[0]
    C[global_C_row_delta+3, global_C_col_base+1] = c_frag_2_3[1]
    C[global_C_row_delta+3, global_C_col_base+2] = c_frag_2_3[2]
    C[global_C_row_delta+3, global_C_col_base+3] = c_frag_2_3[3]
    C[global_C_row_delta+0, global_C_col_delta+0] = c_frag_3_0[0]
    C[global_C_row_delta+0, global_C_col_delta+1] = c_frag_3_0[1]
    C[global_C_row_delta+0, global_C_col_delta+2] = c_frag_3_0[2]
    C[global_C_row_delta+0, global_C_col_delta+3] = c_frag_3_0[3]
    C[global_C_row_delta+1, global_C_col_delta+0] = c_frag_3_1[0]
    C[global_C_row_delta+1, global_C_col_delta+1] = c_frag_3_1[1]
    C[global_C_row_delta+1, global_C_col_delta+2] = c_frag_3_1[2]
    C[global_C_row_delta+1, global_C_col_delta+3] = c_frag_3_1[3]
    C[global_C_row_delta+2, global_C_col_delta+0] = c_frag_3_2[0]
    C[global_C_row_delta+2, global_C_col_delta+1] = c_frag_3_2[1]
    C[global_C_row_delta+2, global_C_col_delta+2] = c_frag_3_2[2]
    C[global_C_row_delta+2, global_C_col_delta+3] = c_frag_3_2[3]
    C[global_C_row_delta+3, global_C_col_delta+0] = c_frag_3_3[0]
    C[global_C_row_delta+3, global_C_col_delta+1] = c_frag_3_3[1]
    C[global_C_row_delta+3, global_C_col_delta+2] = c_frag_3_3[2]
    C[global_C_row_delta+3, global_C_col_delta+3] = c_frag_3_3[3]
    
class Gemm:
   name = 'gemm'
   size = [(i+1) << 8 for i in range(24)]
   gemm_kernel_switcher = {
           1: sgemm_v1,
           2: sgemm_v2,
           3: sgemm_v3,
           4: sgemm_v4,
           5: sgemm_128_128_kernel
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
      ti.profiler.clear_kernel_profiler_info()
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
