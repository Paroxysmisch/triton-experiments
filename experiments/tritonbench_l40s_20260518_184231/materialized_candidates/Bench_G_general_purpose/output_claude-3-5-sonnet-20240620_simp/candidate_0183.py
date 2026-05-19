import torch
import triton
import triton.language as tl
import math

@triton.jit
def softmax_kernel_non_inner(
    output_ptr, input_ptr,
    stride_om, stride_on, stride_ok,
    stride_im, stride_in, stride_ik,
    M, N, K,
    TILE_N: tl.constexpr, TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    # Position of elements processed by this program
    pid = tl.program_id(0)
    if ONE_TILE_PER_CTA:
        pid_m = pid // tl.cdiv(K, TILE_K)
        pid_k = pid % tl.cdiv(K, TILE_K)
    else:
        pid_m = pid // K
        pid_k = pid % K

    # Create offsets for this tile
    offs_k = pid_k * TILE_K + tl.arange(0, TILE_K)
    offs_n = tl.arange(0, TILE_N)
    offs_m = pid_m
    
    # Load input data
    a_ptrs = input_ptr + offs_m * stride_im + offs_n[:, None] * stride_in + offs_k[None, :] * stride_ik
    mask = offs_k[None, :] < K
    row = tl.load(a_ptrs, mask=mask, other=-float('inf'))
    
    # Compute softmax
    row_minus_max = row - tl.max(row, axis=1)[:, None]
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    softmax = numerator / denominator

    # Write output
    out_ptrs = output_ptr + offs_m * stride_om + offs_n[:, None] * stride_on + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, softmax, mask=mask)

@triton.jit
def softmax_kernel_inner(
    output_ptr, input_ptr,
    stride_om, stride_on,
    stride_im, stride_in,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Create block offset
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Load input
    a_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    row = tl.load(a_ptrs, mask=mask, other=-float('inf'))
    
    # Compute softmax
    row_minus_max = row - tl.max(row, axis=1)[:, None]
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    softmax = numerator / denominator
    
    # Store output
    out_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, softmax, mask=mask)

@triton.jit
def softmax_backward_kernel_non_inner(
    grad_input_ptr, grad_output_ptr, output_ptr,
    stride_gim, stride_gin, stride_gik,
    stride_gom, stride_gon, stride_gok,
    stride_om, stride_on, stride_ok,
    M, N, K,
    TILE_N: tl.constexpr, TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid = tl.program_id(0)
    if ONE_TILE_PER_CTA:
        pid_m = pid // tl.cdiv(K, TILE_K)
        pid_k = pid % tl.cdiv(K, TILE_K)
    else:
        pid_m = pid // K
        pid_k = pid % K

    offs_k = pid_k * TILE_K + tl.arange(0, TILE_K)
    offs_n = tl.arange(0, TILE_N)
    offs_m = pid_m
    
    # Load output (softmax result) and grad_output
    y_ptrs = output_ptr + offs_m * stride_om + offs_n[:, None] * stride_on + offs_k[None, :] * stride_ok
    dy_ptrs = grad_output_ptr + offs_m * stride_gom + offs_n[:, None] * stride_gon + offs_k[None, :] * stride_gok
    
    mask = offs_k[None, :] < K
    y = tl.load(y_ptrs, mask=mask, other=0)
    dy = tl.load(dy_ptrs, mask=mask, other=0)
    
    # Compute gradient
    grad = y * (dy - tl.sum(y * dy, axis=1)[:, None])
    
    # Write gradient
    grad_ptrs = grad_input_ptr + offs_m * stride_gim + offs_n[:, None] * stride_gin + offs_k[None, :] * stride_gik
    tl.store(grad_ptrs, grad, mask=mask)

@triton.jit
def softmax_backward_kernel_inner(
    grad_input_ptr, grad_output_ptr, output_ptr,
    stride_gim, stride_gin,
    stride_gom, stride_gon,
    stride_om, stride_on,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Load output and grad_output
    y_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    dy_ptrs = grad_output_ptr + offs_m[:, None] * stride_gom + offs_n[None, :] * stride_gon
    
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y = tl.load(y_ptrs, mask=mask, other=0)
    dy = tl.load(dy_ptrs, mask=mask, other=0)
    
    # Compute gradient
    grad = y * (dy - tl.sum(y * dy, axis=1)[:, None])
    
    # Store gradient
    grad_ptrs = grad_input_ptr + offs_m[:, None] * stride_gim + offs_n[None, :] * stride_gin
    tl.store(grad_ptrs, grad, mask=mask)

def _heur_non_inner(M, N, K):
    TILE_N = min(max(triton.next_power_of_2(N), 32), 128)
    TILE_K = min(max(triton.next_power_of_2(K), 32), 128)
    ONE_TILE_PER_CTA = K <= 8192
    return TILE_N, TILE_K, ONE_TILE_PER_CTA

def _heur_inner(M, N):
    BLOCK_M = min(max(triton.next_power_of_2(M), 32), 256)
    BLOCK_N = min(max(triton.next_power_of_2(N), 32), 256)
    return BLOCK_M, BLOCK_N

class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.stride(-1) == 1:
            M, N = x.shape[-2], x.shape[-1]
            BLOCK_M, BLOCK_N = _heur_inner(M, N)
            output = torch.empty_like(x)
            grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
            softmax_kernel_inner[grid](
                output, x,
                output.stride(-2), output.stride(-1),
                x.stride(-2), x.stride(-1),
                M, N,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
            )
        else:
            *dims, K = x.shape
            x_view = x.view(-1, K)
            M, N = x_view.shape
            TILE_N, TILE_K, ONE_TILE_PER_CTA = _heur_non_inner(M, N, K)
            output = torch.empty_like(x)
            output_view = output.view(-1, K)
            grid = lambda META: (M * (triton.cdiv(K, META["TILE_K"]) if ONE_TILE_PER_CTA else 1),)
            softmax_kernel_non_inner[grid](
                output_view, x_view,
                output_view.stride(0), output_view.stride(1), 1,
                x_view.stride(0), x_view.stride(1), 1,
                M, N, K,
                TILE_N=TILE_N, TILE_K=TILE_K,
                ONE_TILE_PER_CTA=ONE_TILE_PER_CTA
            )
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        if grad_output.stride(-1) == 1:
            M, N = grad_output.shape[-2], grad_output.shape[-1]
            BLOCK_M, BLOCK_N = _heur_inner(M, N)
            grad_input = torch.empty_like(grad_output)
            grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
            softmax_backward_kernel_inner[grid](
                grad_input, grad_output, output,
                grad_input.stride(-2), grad_input.stride(-1),
                grad_output.stride(-2), grad_output.stride(-1),
                output.stride(-2), output.stride(-1),
                M, N,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
            )
        else:
            *dims, K = grad_output.shape
            grad_output_view = grad_output.view(-1, K)
            output_view = output.view(-1, K)
            M, N = grad_output_view.shape
            TILE_N, TILE_K, ONE_TILE_PER_CTA = _heur_non_inner(M, N, K)
            grad_input = torch.empty_like(grad_output)
            grad_input_view = grad_input.view(-1, K)
            grid = lambda META: (M * (triton.cdiv(K, META["TILE_K"]) if ONE_TILE_PER_CTA else 1),)
            softmax_backward_kernel_non_inner[grid](
                grad_input_view, grad_output_view, output_view,
                grad_input_view.stride(0), grad_input_view.stride(1), 1,
                grad_output_view.stride(0), grad_output_view.stride(1), 1,
                output_view.stride(0), output_view.stride(1), 1,
                M, N, K,
                TILE_N=TILE_N, TILE_K=TILE_K,
                ONE_TILE_PER_CTA=ONE_TILE_PER_CTA
            )
        return grad_input
