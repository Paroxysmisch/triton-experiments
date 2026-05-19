import torch
import triton
import triton.language as tl
import math

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

    max_val = tl.max(row, axis=0)
    row -= max_val
    exp_row = tl.exp(row)
    sum_exp = tl.sum(exp_row, axis=0)
    softmax_row = exp_row / sum_exp

    output_row_start = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start + col_offsets
    tl.store(output_ptrs, softmax_row, mask=col_offsets < n_cols)

@triton.jit
def dropout_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols, p,
    seed,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols)

    rng_seed = seed + row_idx
    rng_offset = 0
    random = tl.rand(rng_seed, tl.arange(0, BLOCK_SIZE))
    mask = random > p
    scale = 1.0 / (1.0 - p)
    output_row = tl.where(mask, row * scale, 0.0)

    output_row_start = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start + col_offsets
    tl.store(output_ptrs, output_row, mask=col_offsets < n_cols)

@triton.jit
def layer_norm_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=0.0)

    mean = tl.sum(row, axis=0) / n_cols
    centered = row - mean
    var = tl.sum(centered * centered, axis=0) / n_cols
    std = tl.sqrt(var + eps)
    normalized = centered / std

    output_row_start = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start + col_offsets
    tl.store(output_ptrs, normalized, mask=col_offsets < n_cols)

def fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, *, out=None):
    # Check input compatibility
    assert input.shape[-1] == weight1.size(0), "Input last dim must match weight1's first dim"
    assert weight1.size(1) == weight2.size(0), "weight1's second dim must match weight2's first dim"
    
    original_shape = input.shape
    input_flat = input.reshape(-1, input.size(-1))
    M, K = input_flat.shape
    K, N = weight1.shape
    D_out = weight2.size(1)
    
    # Compute Z1 = input @ weight1
    Z1 = torch.empty((M, N), device=input.device, dtype=input.dtype)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    matmul_kernel[grid](
        input_flat, weight1, Z1,
        M, N, K,
        input_flat.stride(0), input_flat.stride(1),
        weight1.stride(0), weight1.stride(1),
        Z1.stride(0), Z1.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    # Apply softmax
    Z2 = torch.empty_like(Z1)
    BLOCK_SIZE = triton.next_power_of_2(N)
    softmax_kernel[(M,)](
        Z2, Z1,
        Z1.stride(0), Z2.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Apply dropout
    if torch.is_grad_enabled() and dropout_p > 0.0:
        seed = torch.randint(0, 65536, (1,), device='cuda').item()
        Z3 = torch.empty_like(Z2)
        dropout_kernel[(M,)](
            Z3, Z2,
            Z2.stride(0), Z3.stride(0),
            N, dropout_p,
            seed,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        Z3 = Z2
    
    # Compute Z4 = Z3 @ weight2
    Z4 = torch.empty((M, D_out), device=input.device, dtype=input.dtype)
    K2, N2 = weight2.shape
    matmul_kernel[grid](
        Z3, weight2, Z4,
        M, N2, K2,
        Z3.stride(0), Z3.stride(1),
        weight2.stride(0), weight2.stride(1),
        Z4.stride(0), Z4.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    # Reshape and add residual
    Z4_reshaped = Z4.view(*original_shape[:-1], D_out)
    Z4_residual = Z4_reshaped + residual
    
    # Apply LayerNorm
    Z4_flat = Z4_residual.reshape(-1, D_out)
    Y = torch.empty_like(Z4_flat)
    BLOCK_SIZE_LN = triton.next_power_of_2(D_out)
    layer_norm_kernel[(Z4_flat.size(0),)](
        Y, Z4_flat,
        Z4_flat.stride(0), Y.stride(0),
        D_out, eps,
        BLOCK_SIZE=BLOCK_SIZE_LN
    )
    
    output = Y.view(*original_shape[:-1], D_out)
    if out is not None:
        out.copy_(output)
        return out
    return output
