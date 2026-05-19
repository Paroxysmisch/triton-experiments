import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_non_inner(
    output_ptr, input_ptr,
    M, N, K,
    stride_m_out, stride_n_out, stride_k_out,
    stride_m_in, stride_n_in, stride_k_in,
    TILE_K: tl.constexpr, TILE_N: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr
):
    pid = tl.program_id(0)
    if ONE_TILE_PER_CTA:
        m = pid
        n_offset = 0
    else:
        m = pid // (N // TILE_N)
        n_offset = (pid % (N // TILE_N)) * TILE_N
    k_block = tl.arange(0, TILE_K)
    n_block = tl.arange(0, TILE_N) + n_offset
    mask = (k_block < K) & (n_block < N)
    input_offset = (m * stride_m_in + n_block * stride_n_in + k_block * stride_k_in)
    x = tl.load(input_ptr + input_offset, mask=mask, other=-float('inf'))
    x_max = tl.max(x, 0)
    x = x - x_max
    x_exp = tl.exp(x)
    denom = tl.sum(x_exp, 0)
    output_offset = (m * stride_m_out + n_block * stride_n_out + k_block * stride_k_out)
    out = x_exp / denom
    tl.store(output_ptr + output_offset, out, mask=mask)

@triton.jit
def softmax_kernel_inner(
    output_ptr, input_ptr,
    M, N, K,
    stride_m_out, stride_n_out, stride_k_out,
    stride_m_in, stride_n_in, stride_k_in,
    TILE_K: tl.constexpr
):
    pid = tl.program_id(0)
    offset_m = pid // (K // TILE_K)
    offset_k = (pid % (K // TILE_K)) * TILE_K
    k_range = tl.arange(0, TILE_K) + offset_k
    mask = k_range < K
    row_offset = offset_m * stride_m_in
    x = tl.load(input_ptr + row_offset + k_range * stride_k_in, mask=mask, other=-float('inf'))
    x_max = tl.max(x, 0)
    x = x - x_max
    x_exp = tl.exp(x)
    denom = tl.sum(x_exp, 0)
    out = x_exp / denom
    tl.store(output_ptr + offset_m * stride_m_out + k_range * stride_k_out, out, mask=mask)

@triton.jit
def softmax_backward_kernel_non_inner(
    in_grad_ptr, out_ptr, grad_out_ptr,
    M, N, K,
    stride_m_in, stride_n_in, stride_k_in,
    stride_m_out, stride_n_out, stride_k_out,
    stride_m_grad, stride_n_grad, stride_k_grad,
    TILE_K: tl.constexpr, TILE_N: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr
):
    pid = tl.program_id(0)
    if ONE_TILE_PER_CTA:
        m = pid
        n_offset = 0
    else:
        m = pid // (N // TILE_N)
        n_offset = (pid % (N // TILE_N)) * TILE_N
    k_block = tl.arange(0, TILE_K)
    n_block = tl.arange(0, TILE_N) + n_offset
    mask = (k_block < K) & (n_block < N)
    o_offset = (m * stride_m_out + n_block * stride_n_out + k_block * stride_k_out)
    g_offset = (m * stride_m_grad + n_block * stride_n_grad + k_block * stride_k_grad)
    softmax_vals = tl.load(out_ptr + o_offset, mask=mask, other=0.0)
    grad_out_vals = tl.load(grad_out_ptr + g_offset, mask=mask, other=0.0)
    total = tl.sum(softmax_vals * grad_out_vals, 0)
    in_grad = softmax_vals * (grad_out_vals - total)
    i_offset = (m * stride_m_in + n_block * stride_n_in + k_block * stride_k_in)
    tl.store(in_grad_ptr + i_offset, in_grad, mask=mask)

@triton.jit
def softmax_backward_kernel_inner(
    in_grad_ptr, out_ptr, grad_out_ptr,
    M, N, K,
    stride_m_in, stride_n_in, stride_k_in,
    stride_m_out, stride_n_out, stride_k_out,
    stride_m_grad, stride_n_grad, stride_k_grad,
    TILE_K: tl.constexpr
):
    pid = tl.program_id(0)
    offset_m = pid // (K // TILE_K)
    offset_k = (pid % (K // TILE_K)) * TILE_K
    k_range = tl.arange(0, TILE_K) + offset_k
    mask = k_range < K
    row_offset_out = offset_m * stride_m_out
    row_offset_grad = offset_m * stride_m_grad
    softmax_vals = tl.load(out_ptr + row_offset_out + k_range * stride_k_out, mask=mask, other=0.0)
    grad_out_vals = tl.load(grad_out_ptr + row_offset_grad + k_range * stride_k_grad, mask=mask, other=0.0)
    total = tl.sum(softmax_vals * grad_out_vals, 0)
    in_grad = softmax_vals * (grad_out_vals - total)
    row_offset_in = offset_m * stride_m_in
    tl.store(in_grad_ptr + row_offset_in + k_range * stride_k_in, in_grad, mask=mask)

def heur_tile_k(M, N, K):
    if K >= 128:
        return 128
    return 64

def heur_tile_n_non_inner(M, N, K):
    if N >= 128:
        return 128
    return 64

def heur_tile_n_inner(M, N, K):
    return 128

class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, M, N, K):
        output = torch.empty_like(x)
        tile_k = heur_tile_k(M, N, K)
        if K > 1:
            tile_n = heur_tile_n_non_inner(M, N, K)
            grid = lambda META: (M if META['ONE_TILE_PER_CTA'] else (M * ((N + tile_n - 1)//tile_n),),)
            softmax_kernel_non_inner[grid](
                output, x,
                M, N, K,
                x.stride(0), x.stride(1) if x.dim()>1 else 0, x.stride(2) if x.dim()>2 else 0,
                x.stride(0), x.stride(1) if x.dim()>1 else 0, x.stride(2) if x.dim()>2 else 0,
                TILE_K=tile_k, TILE_N=tile_n, ONE_TILE_PER_CTA=True
            )
        else:
            tile = heur_tile_n_inner(M, N, K)
            grid = lambda META: ((M * ((K + tile - 1)//tile)),)
            softmax_kernel_inner[grid](
                output, x,
                M, N, K,
                x.stride(0), x.stride(1) if x.dim()>1 else 0, x.stride(2) if x.dim()>2 else 0,
                x.stride(0), x.stride(1) if x.dim()>1 else 0, x.stride(2) if x.dim()>2 else 0,
                TILE_K=tile
            )
        ctx.save_for_backward(x, output, torch.tensor([M, N, K], device=x.device))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, output, shape_tensor = ctx.saved_tensors
        M, N, K = shape_tensor[0].item(), shape_tensor[1].item(), shape_tensor[2].item()
        grad_input = torch.empty_like(x)
        tile_k = heur_tile_k(M, N, K)
        if K > 1:
            tile_n = heur_tile_n_non_inner(M, N, K)
            grid = lambda META: (M if META['ONE_TILE_PER_CTA'] else (M * ((N + tile_n - 1)//tile_n),),)
            softmax_backward_kernel_non_inner[grid](
                grad_input, output, grad_output,
                M, N, K,
                x.stride(0), x.stride(1) if x.dim()>1 else 0, x.stride(2) if x.dim()>2 else 0,
                output.stride(0), output.stride(1) if output.dim()>1 else 0, output.stride(2) if output.dim()>2 else 0,
                grad_output.stride(0), grad_output.stride(1) if grad_output.dim()>1 else 0, grad_output.stride(2) if grad_output.dim()>2 else 0,
                TILE_K=tile_k, TILE_N=tile_n, ONE_TILE_PER_CTA=True
            )
        else:
            tile = heur_tile_n_inner(M, N, K)
            grid = lambda META: ((M * ((K + tile - 1)//tile)),)
            softmax_backward_kernel_inner[grid](
                grad_input, output, grad_output,
                M, N, K,
                x.stride(0), x.stride(1) if x.dim()>1 else 0, x.stride(2) if x.dim()>2 else 0,
                output.stride(0), output.stride(1) if output.dim()>1 else 0, output.stride(2) if output.dim()>2 else 0,
                grad_output.stride(0), grad_output.stride(1) if grad_output.dim()>1 else 0, grad_output.stride(2) if grad_output.dim()>2 else 0,
                TILE_K=tile
            )
        return grad_input, None, None, None

def softmax_triton(x):
    M, N = x.shape[0], x.shape[1] if x.dim() > 1 else 1
    K = x.shape[2] if x.dim() > 2 else 1
    return Softmax.apply(x, M, N, K)
