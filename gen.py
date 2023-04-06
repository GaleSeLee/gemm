for i in range(8):
    for j in range(8):
        print(f"C[global_idx_c+{i}*N+{j}] = tmp_{i}{j};")
