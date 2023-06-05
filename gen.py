for i in range(4):
    for j in range(4):
        print(f"\t\t\tC[global_C_row_base+{i}, global_C_col_base+{j}] = c_frag_0_{i}[{j}]")

for i in range(4):
    for j in range(4):
        print(f"\t\t\tC[global_C_row_base+{i}, global_C_col_delta+{j}] = c_frag_1_{i}[{j}]")

for i in range(4):
    for j in range(4):
        print(f"\t\t\tC[global_C_row_delta+{i}, global_C_col_base+{j}] = c_frag_2_{i}[{j}]")

for i in range(4):
    for j in range(4):
        print(f"\t\t\tC[global_C_row_delta+{i}, global_C_col_delta+{j}] = c_frag_3_{i}[{j}]")
