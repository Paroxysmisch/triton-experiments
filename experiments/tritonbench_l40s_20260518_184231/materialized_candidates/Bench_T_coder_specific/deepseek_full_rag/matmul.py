import torch
import triton
import triton.language as tl

@triton.jit
def matmul_1d_1d_kernel(a_ptr, b_ptr, c_ptr, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """
    Compute the matrix multiplication of a 1D tensor with shape (M, ) and a 1D tensor with shape (N, ) and store the result in a 1D tensor with shape (M, ).
    """
    pid = tl.program_id(axis=0)
    a_pid = pid
    b_pid = pid
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = a_pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    a_pid = first_pid_m + (a_pid % group_size_m)
    a_row = a_pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    a_mask = a_row < M
    b_col = b_pid % N
    b_row = (b_pid // N) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    b_mask = b_row < N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, num_pid_n):
        a = tl.load(a_ptr + a_row * M + a_col, mask=a_mask[None] & (a_col < M))
        b = tl.load(b_ptr + b_row * N + b_col, mask=b_mask[None] & (b_row < N))
        accumulator += a[:, None] * b[None, :]
        b_pid += num_pid_m
        b_row = (b_pid // N) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        b_mask = b_row < N

    c = tl.sum(accumulator, axis=1)
    c_row = a_pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_mask = c_row < M
    tl.store(c_ptr + c_row * M + c_col, c, mask=c_mask[None] & (c_col < M))

@triton.jit
def matmul_1d_2d_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    """
    Compute the matrix multiplication of a 1D tensor with shape (M, ) and a 2D tensor with shape (N, K) and store the result in a 1D tensor with shape (M, ).
    """
    pid = tl.program_id(axis=0)
    a_pid = pid
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N * K, BLOCK_SIZE_N * BLOCK_SIZE_K)
    n = tl.arange(0, BLOCK_SIZE_N)
    k = tl.arange(0, BLOCK_SIZE_K)
    a_row = a_pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    a_col = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.int32)
    a_mask = (a_row < M)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    for _ in range(0, num_pid_n):
        b_pid = pid
        b_row = b_pid * BLOCK_SIZE_N + n
        b_col = b_pid * BLOCK_SIZE_K + k
        b_mask = (b_row < N * K)
        b = tl.load(b_ptr + b_row * K + b_col, mask=b_mask[None, None] & (b_col < K)).to(tl.float32)
        a = tl.load(a_ptr + a_row * K + a_col, mask=a_mask[None, None] & (a_col < K)).to(tl.float32)
        accumulator += a[:, None, None] * b[None, :, :]
        b_pid += num_pid_m
        b_row = b_pid * BLOCK_SIZE_N + n
        b_col = b_pid * BLOCK_SIZE_K + k
        b_mask = (b_row < N * K)
    c = tl.sum(accumulator, axis=(1, 2))
    c_row = a_pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_mask = (c_row < M)
    tl.store(c_ptr + c_row, c, mask=c_mask)

@triton.jit
def matmul_batched_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """
    Compute the batched matrix multiplication of two 2D tensors and store the result in a 3D tensor.
    """
    pid = tl.program_id(axis=0)
    a_pid = pid
    b_pid = pid
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n * num_pid_k
    group_id = a_pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    a_pid = first_pid_m + (a_pid % group_size_m)
    a_row = a_pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    a_col = (a_pid // num_pid_n) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_mask = (a_row < M) & (a_col < K)
    b_row = (b_pid % num_pid_m) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    b_col = (b_pid // num_pid_k) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    b_mask = (b_row < M) & (b_col < K)
    c_row = a_pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_col = (a_pid // num_pid_k) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (c_row <
