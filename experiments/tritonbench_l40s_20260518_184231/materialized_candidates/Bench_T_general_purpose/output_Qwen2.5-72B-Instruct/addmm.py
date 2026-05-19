import triton
import triton.language as tl

@triton.jit
def addmm_kernel(
    input_ptr, mat1_ptr, mat2_ptr, out_ptr,
    input_m, input_n, mat1_m, mat2_n, mat1_k,
    stride_input_m, stride_input_n,
    stride_mat1_m, stride_mat1_k,
    stride_mat2_k, stride_mat2_n,
    stride_out_m, stride_out_n,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(input_m, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(input_n, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % input_m
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % input_n
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = mat1_ptr + (offs_am[:, None] * stride_mat1_m + offs_k[None, :] * stride_mat1_k)
    b_ptrs = mat2_ptr + (offs_k[:, None] * stride_mat2_k + offs_bn[None, :] * stride_mat2_n)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, mat1_k, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_mat1_k
        b_ptrs += BLOCK_SIZE_K * stride_mat2_k

    accumulator = accumulator * alpha

    if beta != 0:
        c_ptrs = input_ptr + (offs_am[:, None] * stride_input_m + offs_bn[None, :] * stride_input_n)
        c_mask = (offs_am[:, None] < input_m) & (offs_bn[None, :] < input_n)
        c = tl.load(c_ptrs, mask=c_mask, other=0.0)
        accumulator += c * beta

    out_ptrs = out_ptr + (offs_am[:, None] * stride_out_m + offs_bn[None, :] * stride_out_n)
    out_mask = (offs_am[:, None] < input_m) & (offs_bn[None, :] < input_n)
    tl.store(out_ptrs, accumulator, mask=out_mask)

import torch
import triton
import triton.language as tl

def addmm(input, mat1, mat2, beta=1, alpha=1, out=None):
    assert input.dim() == 2, "input must be a 2D tensor"
    assert mat1.dim() == 2, "mat1 must be a 2D tensor"
    assert mat2.dim() == 2, "mat2 must be a 2D tensor"
    assert mat1.size(1) == mat2.size(0), "mat1 and mat2 dimensions must be compatible for multiplication"

    input_m, input_n = input.size()
    mat1_m, mat1_k = mat1.size()
    mat2_k, mat2_n = mat2.size()
    assert input_m == mat1_m and input_n == mat2_n, "input dimensions must match the result dimensions"

    if out is None:
        out = torch.empty_like(input)

    assert out.size() == input.size(), "out tensor must have the same shape as input"

    grid = lambda META: (
        triton.cdiv(input_m, META['BLOCK_SIZE_M']) * triton.cdiv(input_n, META['BLOCK_SIZE_N']),
    )

    addmm_kernel[grid](
        input, mat1, mat2, out,
        input_m, input_n, mat1_m, mat2_n, mat1_k,
        input.stride(0), input.stride(1),
        mat1.stride(0), mat1.stride(1),
        mat2.stride(0), mat2.stride(1),
        out.stride(0), out.stride(1),
        alpha, beta,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16,
        GROUP_SIZE_M=8
    )

    return out
