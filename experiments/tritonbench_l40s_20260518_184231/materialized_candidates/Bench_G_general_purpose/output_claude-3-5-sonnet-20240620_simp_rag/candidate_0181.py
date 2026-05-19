import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def softmax_kernel_non_inner(
    output_ptr, input_ptr,
    M, N, K,
    stride_om, stride_on, stride_ok,
    stride_im, stride_in, stride_ik,
    TILE_K: tl.constexpr, TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute tile indices
    tile_k = pid // (M * tl.cdiv(N, TILE_N)) if not ONE_TILE_PER_CTA else 0
    tile_mn = pid % (M * tl.cdiv(N, TILE_N)) if not ONE_TILE_PER_CTA else pid
    tile_m = tile_mn // tl.cdiv(N, TILE_N)
    tile_n = tile_mn % tl.cdiv(N, TILE_N)

    # Initialize offsets
    offs_k = tl.arange(0, TILE_K)
    offs_n = tl.arange(0, TILE_N)
    
    # Compute input/output pointers
    input_ptr = input_ptr + tile_k * stride_ik + tile_m * stride_im + tile_n * stride_in
    output_ptr = output_ptr + tile_k * stride_ok + tile_m * stride_om + tile_n * stride_on

    # Load input block
    x = tl.load(input_ptr + offs_k[:, None] * stride_ik + offs_n[None, :] * stride_in,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N - tile_n * TILE_N),
                other=-float('inf'))
    
    # Compute max for numerical stability
    x_max = tl.max(x, 0)
    x = x - x_max[None, :]
    
    # Compute exponentials and sum
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, 0)
    
    # Compute softmax
    softmax_output = numerator / denominator[None, :]
    
    # Store result
    tl.store(output_ptr + offs_k[:, None] * stride_ok + offs_n[None, :] * stride_on,
             softmax_output,
             mask=(offs_k[:, None] < K) & (offs_n[None, :] < N - tile_n * TILE_N))

@triton.jit
def softmax_kernel_inner(
    output_ptr, input_ptr,
    M, N,
    stride_om, stride_on,
    stride_im, stride_in,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Load row into SRAM
    row_start_ptr = input_ptr + pid * stride_im
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row = tl.load(row_start_ptr + col_offsets * stride_in,
                  mask=col_offsets < N,
                  other=-float('inf'))
    
    # Compute softmax
    row_max = tl.max(row, axis=0)
    row = row - row_max
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # Store result
    output_row_ptr = output_ptr + pid * stride_om
    tl.store(output_row_ptr + col_offsets * stride_on,
             softmax_output,
             mask=col_offsets < N)

# Heuristic functions for kernel configuration
def heur_num_warps(N):
    if N <= 512:
        return 4
    if N <= 2048:
        return 8
    return 16

def heur_tile_size(N, K):
    if N <= 256:
        return min(N, 64), min(K, 64)
    if N <= 1024:
        return min(N, 128), min(K, 128)
    return min(N, 256), min(K, 256)

class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save input for backward pass
        ctx.save_for_backward(x)
        
        # Get input dimensions
        *shape, N = x.shape
        M = prod(shape) if shape else 1
        
        # Allocate output
        output = torch.empty_like(x)
        
        if len(shape) <= 1:
            # Use inner kernel for 1D/2D case
            BLOCK_SIZE = triton.next_power_of_2(N)
            num_warps = heur_num_warps(N)
            
            softmax_kernel_inner[(M,)](
                output, x,
                M, N,
                output.stride(-2), output.stride(-1),
                x.stride(-2), x.stride(-1),
                num_warps=num_warps,
                BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            # Use non-inner kernel for higher dimensions
            K = shape[-1]
            TILE_N, TILE_K = heur_tile_size(N, K)
            
            grid = (M * triton.cdiv(N, TILE_N) * triton.cdiv(K, TILE_K),)
            softmax_kernel_non_inner[grid](
                output, x,
                M, N, K,
                output.stride(-3), output.stride(-2), output.stride(-1),
                x.stride(-3), x.stride(-2), x.stride(-1),
                TILE_K=TILE_K, TILE_N=TILE_N,
                ONE_TILE_PER_CTA=K <= TILE_K
            )
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * (softmax(x) * (1 - softmax(x)))

def softmax(x: torch.Tensor) -> torch.Tensor:
    return Softmax.apply(x)
