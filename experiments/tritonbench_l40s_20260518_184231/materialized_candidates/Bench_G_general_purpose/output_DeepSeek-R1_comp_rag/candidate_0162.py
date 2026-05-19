import torch
import triton
import triton.language as tl

# Forward Kernels
@triton.jit
def softmax_kernel_non_inner(
    output_ptr, input_ptr,
    M, N, K,
    stride_zm, stride_zn, stride_zk,
    stride_im, stride_in, stride_ik,
    TILE_K: tl.constexpr, TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * TILE_K + tl.arange(0, TILE_K)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    
    input_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator
    
    output_ptrs = output_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn
    tl.store(output_ptrs, softmax_out, mask=mask)

@triton.jit
def softmax_kernel_inner(
    output_ptr, input_ptr,
    M, N, K,
    stride_zm, stride_zn, stride_zk,
    stride_im, stride_in, stride_ik,
    TILE_K: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, TILE_K)
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_m
    
    offs_m = pid_m * TILE_K + tl.arange(0, TILE_K)
    offs_n = pid_n * TILE_K + tl.arange(0, TILE_K)
    
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    input_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    row_minus_max = row - tl.max(row, axis=1, keepdims=True)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1, keepdims=True)
    softmax_out = numerator / denominator
    
    output_ptrs = output_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn
    tl.store(output_ptrs, softmax_out, mask=mask)

# Backward Kernels
@triton.jit
def softmax_backward_kernel_non_inner(
    in_grad_ptr, output_ptr, grad_ptr,
    M, N, K,
    stride_gm, stride_gn, stride_gk,
    stride_zm, stride_zn, stride_zk,
    stride_gradm, stride_gradn, stride_gradk,
    TILE_K: tl.constexpr, TILE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * TILE_K + tl.arange(0, TILE_K)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    grad_ptrs = grad_ptr + offs_m[:, None] * stride_gradm + offs_n[None, :] * stride_gradn
    grad = tl.load(grad_ptrs, mask=mask, other=0.0)
    
    out_ptrs = output_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn
    output = tl.load(out_ptrs, mask=mask, other=0.0)
    
    sum_term = tl.sum(output * grad, axis=1, keepdims=True)
    grad_input = output * (grad - sum_term)
    
    in_grad_ptrs = in_grad_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    tl.store(in_grad_ptrs, grad_input, mask=mask)

@triton.jit
def softmax_backward_kernel_inner(
    in_grad_ptr, output_ptr, grad_ptr,
    M, N, K,
    stride_gm, stride_gn, stride_gk,
    stride_zm, stride_zn, stride_zk,
    stride_gradm, stride_gradn, stride_gradk,
    TILE_K: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, TILE_K)
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_m
    
    offs_m = pid_m * TILE_K + tl.arange(0, TILE_K)
    offs_n = pid_n * TILE_K + tl.arange(0, TILE_K)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    grad = tl.load(grad_ptr + offs_m[:, None] * stride_gradm + offs_n[None, :] * stride_gradn, 
                   mask=mask, other=0.0)
    output = tl.load(output_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn, 
                     mask=mask, other=0.0)
    
    sum_term = tl.sum(output * grad, axis=1, keepdims=True)
    grad_input = output * (grad - sum_term)
    
    tl.store(in_grad_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn, 
            grad_input, mask=mask)

# Heuristic Functions
def heur_tile_k(M, N, K):
    if K <= 2048: return 128
    return 64 if K <= 4096 else 32

def heur_tile_n_non_inner(M, N, K):
    if N <= 512: return 64
    return 32 if N <= 1024 else 16

def heur_tile_n_inner(M, N, K):
    return 128 if K <= 1024 else 64

# Softmax Autograd Function
class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=-1):
        dim = dim if dim >= 0 else x.dim() + dim
        ctx.save_for_backward(x)
        ctx.dim = dim
        
        M, N, K = x.shape[0], x.shape[1], x.shape[2] if x.dim() > 2 else 1
        is_inner = dim == x.dim() - 1
        
        if is_inner:
            TILE_K = heur_tile_k(M, N, K)
            grid = lambda opt: (triton.cdiv(M, opt['TILE_K']) * triton.cdiv(N, opt['TILE_K']),)
            softmax_kernel_inner[grid](
                x, x, M, N, K,
                x.stride(0), x.stride(1), 0,
                x.stride(0), x.stride(1), 0,
                TILE_K=TILE_K
            )
        else:
            TILE_K = heur_tile_k(M, N, K)
            TILE_N = heur_tile_n_non_inner(M, N, K)
            grid = lambda opt: (triton.cdiv(M, opt['TILE_K']), triton.cdiv(N, opt['TILE_N']))
            softmax_kernel_non_inner[grid](
                x, x, M, N, K,
                x.stride(0), x.stride(1), 0,
                x.stride(0), x.stride(1), 0,
                TILE_K=TILE_K, TILE_N=TILE_N,
                ONE_TILE_PER_CTA=1
            )
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        dim = ctx.dim
        grad_input = torch.empty_like(x)
        
        M, N, K = x.shape[0], x.shape[1], x.shape[2] if x.dim() > 2 else 1
        is_inner = dim == x.dim() - 1
        
        if is_inner:
            TILE_K = heur_tile_k(M, N, K)
            grid = lambda opt: (triton.cdiv(M, opt['TILE_K']) * triton.cdiv(N, opt['TILE_K']),)
            softmax_backward_kernel_inner[grid](
                grad_input, x, grad_output,
                M, N, K,
                grad_input.stride(0), grad_input.stride(1), 0,
                x.stride(0), x.stride(1), 0,
                grad_output.stride(0), grad_output.stride(1), 0,
                TILE_K=TILE_K
            )
        else:
            TILE_K = heur_tile_k(M, N, K)
            TILE_N = heur_tile_n_non_inner(M, N, K)
            grid = lambda opt: (triton.cdiv(M, opt['TILE_K']), triton.cdiv(N, opt['TILE_N']))
            softmax_backward_kernel_non_inner[grid](
                grad_input, x, grad_output,
                M, N, K,
                grad_input.stride(0), grad_input.stride(1), 0,
                x.stride(0), x.stride(1), 0,
                grad_output.stride(0), grad_output.stride(1), 0,
                TILE_K=TILE_K, TILE_N=TILE_N
            )
        return grad_input, None

# Wrapper Function
def softmax(x, dim=-1):
    return Softmax.apply(x, dim)
