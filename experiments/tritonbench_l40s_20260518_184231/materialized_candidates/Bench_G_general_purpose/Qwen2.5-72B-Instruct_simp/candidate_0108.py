import triton
import triton.language as tl

@triton.jit
def kernel(M, Out, stride_m_row, stride_m_col, stride_out_row, stride_out_col, SIZE_M, D_HEAD, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = SIZE_M // BLOCK_SIZE
    num_pid_n = D_HEAD // BLOCK_SIZE
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    first_pid_n = group_id * num_pid_n
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = first_pid_n + (pid % num_pid_n)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE), BLOCK_SIZE)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE), BLOCK_SIZE)

    m_mask = offs_m < SIZE_M
    n_mask = offs_n < D_HEAD

    offs = offs_m[:, None] * stride_m_row + offs_n[None, :] * stride_m_col
    In = tl.load(M + offs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)

    offs = offs_n[:, None] * stride_out_row + offs_m[None, :] * stride_out_col
    tl.store(Out + offs, In, mask=m_mask[None, :] & n_mask[:, None])

import torch
import triton
import triton.language as tl

def transpose_matrix(M, D_HEAD, SIZE_M, BLOCK_SIZE=128):
    # Initialize the input matrix with random values
    M = torch.randn((SIZE_M, D_HEAD), device='cuda', dtype=torch.float32)
    
    # Initialize the output matrix with zeros
    Out = torch.zeros((D_HEAD, SIZE_M), device='cuda', dtype=torch.float32)
    
    # Compute the strides
    stride_m_row = M.stride(0)
    stride_m_col = M.stride(1)
    stride_out_row = Out.stride(0)
    stride_out_col = Out.stride(1)
    
    # Launch the kernel
    grid = lambda meta: (triton.cdiv(SIZE_M * D_HEAD, meta['BLOCK_SIZE'] * meta['BLOCK_SIZE']),)
    kernel[grid](M, Out, stride_m_row, stride_m_col, stride_out_row, stride_out_col, SIZE_M, D_HEAD, BLOCK_SIZE)
    
    return Out

# Example usage
SIZE_M = 1024
D_HEAD = 1024
transposed_matrix = transpose_matrix(None, D_HEAD, SIZE_M)
print(transposed_matrix)
