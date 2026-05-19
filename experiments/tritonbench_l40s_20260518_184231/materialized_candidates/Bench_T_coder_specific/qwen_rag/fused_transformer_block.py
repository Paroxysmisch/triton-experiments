import torch
import triton
import triton.language as tl

# Matrix Multiplication Kernel
@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, A_row_stride, A_col_stride, B_row_stride, B_col_stride, C_row_stride, C_col_stride, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_blocks_M = tl.cdiv(M, BLOCK_SIZE_M)
    num_blocks_N = tl.cdiv(N, BLOCK_SIZE_N)
    group_size_m = num_blocks_M * BLOCK_SIZE_M
    group_size_n = num_blocks_N * BLOCK_SIZE_N
    group_idx = pid // group_size_n
    local_pid = pid % group_size_n

    A_base_ptr = A_ptr + group_idx * A_row_stride * BLOCK_SIZE_M * K
    B_base_ptr = B_ptr + local_pid * B_col_stride * K * BLOCK_SIZE_N
    C_base_ptr = C_ptr + group_idx * C_row_stride * BLOCK_SIZE_M * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        A_shared = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        B_shared = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
        A_shared[:, :K] = tl.load(A_base_ptr + pid % group_size_m * A_col_stride * K + k * A_col_stride, mask=(pid % group_size_m * BLOCK_SIZE_M + tl.arange(BLOCK_SIZE_M)) < M and k + tl.arange(BLOCK_SIZE_K) < K, other=float('-inf'))
        B_shared[:K, :] = tl.load(B_base_ptr + (local_pid // BLOCK_SIZE_N) * B_row_stride * K * BLOCK_SIZE_N + (local_pid % BLOCK_SIZE_N) * B_col_stride + k * B_col_stride, mask=(k + tl.arange(BLOCK_SIZE_K) < K and (local_pid // BLOCK_SIZE_N) * BLOCK_SIZE_N + tl.arange(BLOCK_SIZE_N) < N), other=float('-inf'))
        accumulator += tl.dot(A_shared, B_shared)
    tl.store(C_base_ptr + pid % group_size_m * C_col_stride * BLOCK_SIZE_N + (local_pid // BLOCK_SIZE_N) * C_row_stride, accumulator.to(C_ptr.dtype), mask=(pid % group_size_m * BLOCK_SIZE_M + tl.arange(BLOCK_SIZE_M) < M and (local_pid // BLOCK_SIZE_N) * BLOCK_SIZE_N + tl.arange(BLOCK_SIZE_N) < N))

# Softmax Kernel
@triton.jit
def softmax_kernel(X_ptr, Y_ptr, X_row_stride, Y_row_stride, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_blocks = tl.cdiv(N, BLOCK_SIZE)
    row_start_ptr = X_ptr + pid * X_row_stride
    row_start_ptr_y = Y_ptr + pid * Y_row_stride
    max_val = -float('inf')
    for i in range(BLOCK_SIZE):
        val = tl.load(row_start_ptr + i, mask=i < N, other=-float('inf'))
        max_val = tl.max(max_val, val)
    tl.sync_mem()
    exp_sum = 0.0
    for i in range(BLOCK_SIZE):
        val = tl.exp(tl.load(row_start_ptr + i, mask=i < N, other=-float('inf')) - max_val)
        exp_sum += val
    tl.sync_mem()
    inv_exp_sum = 1.0 / exp_sum
    for i in range(BLOCK_SIZE):
        val = tl.exp(tl.load(row_start_ptr + i, mask=i < N, other=-float('inf')) - max_val) * inv_exp_sum
        tl.store(row_start_ptr_y + i, val, mask=i < N)

# Dropout Kernel
@triton.jit
def dropout_kernel(X_ptr, Y_ptr, X_row_stride, Y_row_stride, N, P, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_blocks = tl.cdiv(N, BLOCK_SIZE)
    row_start_ptr = X_ptr + pid * X_row_stride
    row_start_ptr_y = Y_ptr + pid * Y_row_stride
    for i in range(BLOCK_SIZE):
        val = tl.load(row_start_ptr + i, mask=i < N, other=0.0)
        if tl.random() < P:
            val = 0.0
        tl.store(row_start_ptr_y + i, val, mask=i < N)

# Layer Normalization Kernel
@triton.jit
def layernorm_kernel(X_ptr, Y_ptr, gamma_ptr, beta_ptr, X_row_stride, Y_row_stride, N, EPS, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_blocks = tl.cdiv(N, BLOCK_SIZE)
    row_start_ptr = X_ptr + pid * X_row_stride
    row_start_ptr_y = Y_ptr + pid * Y_row_stride
    sum_val = 0.0
    sq_sum_val = 0.0
    for i in range(BLOCK_SIZE):
        val = tl.load(row_start_ptr + i, mask=i < N, other=0.0)
        sum_val += val
        sq_sum_val += val * val
    sum_val /= N
    sq_sum_val /= N
    var = sq_sum_val - sum_val * sum_val + EPS
    inv_var = 1.0 / tl.sqrt(var)
    for i in range(BLOCK_SIZE):
        val = tl.load(row_start_ptr + i, mask=i < N, other=0.0)
        norm_val = (val - sum_val) * inv_var
        scaled_val = norm_val * gamma_ptr[pid % BLOCK_SIZE] + beta_ptr[pid % BLOCK_SIZE]
        tl.store(row_start_ptr_y + i, scaled_val, mask=i < N)

# Triton Wrapper Function
def fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, *, out=None):
    N, D_in = input.shape[-2:]
    D_k, D_out = weight1.shape
    assert D_in == weight1.shape[0], "Incompatible dimensions"
    assert D_k == weight2.shape[0], "Incompatible dimensions"
    assert D_out == weight2.shape[1], "Incompatible dimensions"

    output = torch.empty_like(input)
    intermediate = torch.empty((*input.shape[:-2], D_k), dtype=input.dtype, device=input.device)
    dropout_mask = torch.empty((*input.shape[:-2], D_k), dtype=torch.bool, device=input.device)

    # Z1 = X @ W1
    matmul_kernel[input.shape[0], 1](input.data_ptr(), weight1.data_ptr(), intermediate.data_ptr(), input.stride(0), input.stride(1), weight1.stride(0), weight1.stride(1), intermediate.stride(0), intermediate.stride(1), N, D_k, D_in, BLOCK_SIZE_M=32, BLOCK_SIZE_K=32, BLOCK_SIZE_N=32)

    # Z2 = softmax(Z1)
    softmax_kernel[intermediate.shape[0], 1](intermediate.data_ptr(), intermediate.data_ptr(), intermediate.stride(0), intermediate.stride(1), D_k, BLOCK_SIZE=32)

    # Z3 = dropout(Z2, p)
    dropout_kernel[intermediate.shape[0], 1](intermediate.data_ptr(), dropout_mask.data_ptr(), intermediate.stride(0), dropout_mask.stride(0), D_k, dropout_p, BLOCK_SIZE=32)

    # Z4 = Z3 @ W2
    matmul_kernel[intermediate.shape[0], 1](intermediate.data_ptr(), weight2.data_ptr(), output.data_ptr(), intermediate.stride(0), intermediate.stride(1), weight2.stride(0), weight2.stride(1), output.stride(0), output.stride(1), N, D_out, D_k, BLOCK_SIZE_M=32, BLOCK_SIZE_K=32, BLOCK_SIZE_N=32)

    # Y = LayerNorm(Z4 + R, gamma, beta, epsilon)
    layer_norm_kernel[output.shape[0], 1](output.data_ptr(), output.data_ptr(), torch.ones_like(output), torch.zeros_like(output), output.stride(0), output.stride(1), D_out, eps, BLOCK_SIZE=32)

    return output

# Example usage
if __name__ == "__main__":
    input = torch.randn(1, 128, 768, device='cuda')
    weight1 = torch.randn(768, 2048, device='cuda')
    weight2 = torch.randn(2048, 768, device='cuda')
    residual = torch.randn(1, 128, 76
