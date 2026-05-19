import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_multiply_and_row_dot_kernel(
    A_ptr, B_ptr, C_ptr, alpha, beta, M, N, K, stride_Am, stride_An, stride_Bm, stride_Bn, stride_Cm, stride_Cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    first_pid_n = group_id * num_pid_n
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = first_pid_n + (pid % num_pid_n)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_am[:, None] * stride_Am + offs_k[None, :] * stride_An)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_Bm + offs_bn[None, :] * stride_Bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a_mask = (offs_k[None, :] < K - k)
        b_mask = (offs_k[:, None] < K - k)
        a = tl.load(a_ptrs, mask=a_mask, other=0.)
        b = tl.load(b_ptrs, mask=b_mask, other=0.)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_An
        b_ptrs += BLOCK_SIZE_K * stride_Bm

    accumulator = accumulator * alpha
    c_ptrs = C_ptr + (offs_am[:, None] * stride_Cm + offs_bn[None, :] * stride_Cn)
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    c = tl.load(c_ptrs, mask=c_mask, other=0.)
    c = c * beta + accumulator
    tl.store(c_ptrs, c, mask=c_mask)

    # Compute dot product of the first two rows
    if pid_m == 0 and pid_n == 0:
        row1 = tl.load(c_ptrs + 0 * stride_Cm, mask=offs_bn < N, other=0.)
        row2 = tl.load(c_ptrs + 1 * stride_Cm, mask=offs_bn < N, other=0.)
        dot_product = tl.sum(row1 * row2)
        tl.store(C_ptr + M * N, dot_product)

### PyTorch Wrapper Function

The PyTorch wrapper function will handle the input and output tensors and call the Triton kernel.
