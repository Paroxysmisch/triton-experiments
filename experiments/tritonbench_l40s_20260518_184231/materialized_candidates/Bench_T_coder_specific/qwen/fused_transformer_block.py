@triton.jit
def matmul_kernel(X_ptr, W_ptr, Z1_ptr, M, N, K):
    row = triton.program_id(0)
    col = triton.program_id(1)
    x_offset = row * K + col
    w_offset = col * N + row
    z1_offset = row * N + col
    x_value = X_ptr[x_offset]
    w_value = W_ptr[w_offset]
    Z1_ptr[z1_offset] += x_value * w_value
