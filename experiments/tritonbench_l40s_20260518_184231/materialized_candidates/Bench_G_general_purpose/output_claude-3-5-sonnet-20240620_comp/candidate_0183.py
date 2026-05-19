import torch
import triton
import triton.language as tl

# Heuristic functions for tile size selection
@triton.heuristics({'ONE_TILE_PER_CTA': lambda args: args['K'] <= 64})
def heur_tile_k(K):
    if K <= 32:
        return 32
    if K <= 64:
        return 64
    if K <= 128:
        return 128
    return 256

def heur_tile_n_non_inner(N):
    if N <= 32:
        return 32
    if N <= 64:
        return 64
    return 128

# Forward kernel for non-inner dimensions
@triton.jit
def softmax_kernel_non_inner(
    output_ptr, input_ptr,
    stride_om, stride_on, stride_ok,
    stride_im, stride_in, stride_ik,
    M, N, K,
    TILE_K: tl.constexpr, TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, TILE_K)
    num_pid_n = tl.cdiv(N, TILE_N)
    
    # Calculate indices
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Initialize offsets
    offs_m = pid_m * TILE_K + tl.arange(0, TILE_K)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, K)
    
    # Create mask
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Initialize maximum values
    max_val = tl.full([TILE_K, TILE_N], float('-inf'), dtype=tl.float32)
    
    # Load input and compute max
    for k in range(0, K):
        x = tl.load(input_ptr + offs_m[:, None] * stride_im + 
                   offs_n[None, :] * stride_in + k * stride_ik,
                   mask=mask, other=float('-inf'))
        max_val = tl.maximum(max_val, x)
    
    # Compute exponentials and sum
    numerator = tl.zeros([TILE_K, TILE_N], dtype=tl.float32)
    denominator = tl.zeros([TILE_K, TILE_N], dtype=tl.float32)
    
    for k in range(0, K):
        x = tl.load(input_ptr + offs_m[:, None] * stride_im +
                   offs_n[None, :] * stride_in + k * stride_ik,
                   mask=mask, other=float('-inf'))
        x = tl.exp(x - max_val)
        numerator += x
        denominator += x
    
    # Compute softmax
    output = numerator / (denominator + 1e-6)
    
    # Store result
    tl.store(output_ptr + offs_m[:, None] * stride_om +
             offs_n[None, :] * stride_on,
             output, mask=mask)

# Forward kernel for inner dimension
@triton.jit
def softmax_kernel_inner(
    output_ptr, input_ptr,
    stride_om, stride_on,
    stride_im, stride_in,
    M, N,
    TILE_N: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Calculate row index
    row = pid
    
    # Initialize offsets
    offs_n = tl.arange(0, TILE_N)
    mask = offs_n < N
    
    # Load input row
    row_ptr = input_ptr + row * stride_im
    x = tl.load(row_ptr + offs_n * stride_in, mask=mask, other=float('-inf'))
    
    # Compute max for numerical stability
    max_val = tl.max(x, axis=0)
    
    # Compute exponentials and sum
    z = tl.exp(x - max_val)
    sum_z = tl.sum(z, axis=0) + 1e-6
    
    # Compute softmax
    out = z / sum_z
    
    # Store result
    out_ptr = output_ptr + row * stride_om
    tl.store(out_ptr + offs_n * stride_on, out, mask=mask)
