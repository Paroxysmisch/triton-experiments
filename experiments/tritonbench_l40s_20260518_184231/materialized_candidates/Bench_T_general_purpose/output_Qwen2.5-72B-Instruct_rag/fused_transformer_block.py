import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    A = A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A, mask=k + rk < K, other=0.0)
        b = tl.load(B, mask=k + rk < K, other=0.0)
        acc += tl.dot(a, b)
    C = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(C, acc, mask=rm[:, None] < M & rn[None, :] < N)

@triton.jit
def softmax_kernel(
    X_ptr, Y_ptr,
    N,
    stride_x, stride_y,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = X_ptr + row_idx * stride_x
    col_offsets = tl.arange(0, BLOCK_SIZE)
    X = tl.load(row_start_ptr + col_offsets, mask=col_offsets < N, other=-float('inf'))
    max_val = tl.max(X, axis=0)
    X = X - max_val
    exp_X = tl.exp(X)
    sum_exp_X = tl.sum(exp_X, axis=0)
    softmax_X = exp_X / sum_exp_X
    Y = Y_ptr + row_idx * stride_y
    tl.store(Y + col_offsets, softmax_X, mask=col_offsets < N)

@triton.jit
def dropout_kernel(
    X_ptr, Y_ptr, dropout_p,
    N,
    stride_x, stride_y,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = X_ptr + row_idx * stride_x
    col_offsets = tl.arange(0, BLOCK_SIZE)
    X = tl.load(row_start_ptr + col_offsets, mask=col_offsets < N, other=0.0)
    mask = tl.rand() > dropout_p
    Y = X * mask / (1 - dropout_p)
    Y_ptr = Y_ptr + row_idx * stride_y
    tl.store(Y_ptr + col_offsets, Y, mask=col_offsets < N)

@triton.jit
def layer_norm_kernel(
    X_ptr, Y_ptr, gamma_ptr, beta_ptr, eps,
    N,
    stride_x, stride_y, stride_gamma, stride_beta,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = X_ptr + row_idx * stride_x
    col_offsets = tl.arange(0, BLOCK_SIZE)
    X = tl.load(row_start_ptr + col_offsets, mask=col_offsets < N, other=0.0)
    mean = tl.sum(X, axis=0) / N
    var = tl.sum((X - mean) * (X - mean), axis=0) / N
    norm_X = (X - mean) / tl.sqrt(var + eps)
    gamma = tl.load(gamma_ptr + col_offsets, mask=col_offsets < N, other=1.0)
    beta = tl.load(beta_ptr + col_offsets, mask=col_offsets < N, other=0.0)
    Y = norm_X * gamma + beta
    Y_ptr = Y_ptr + row_idx * stride_y
    tl.store(Y_ptr + col_offsets, Y, mask=col_offsets < N)
