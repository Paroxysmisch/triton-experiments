import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_non_inner(
    output_ptr, input_ptr, M, N, K,
    stride_m, stride_n, stride_k,
    TILE_K: tl.constexpr, TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    if pid_m >= M:
        return
    
    n_start = pid_n * TILE_N
    n_offs = n_start + tl.arange(0, TILE_N)
    mask_n = n_offs < N
    
    offs_m_base = pid_m * stride_m + n_offs * stride_n
    
    max_val = tl.full((TILE_N,), float('-inf'), dtype=tl.float32)
    sum_val = tl.zeros((TILE_N,), dtype=tl.float32)
    
    for k in range(0, K, TILE_K):
        k_offs = k + tl.arange(0, TILE_K)
        mask_k = k_offs < K
        
        input_ptrs = input_ptr + offs_m_base[:, None] + k_offs[None, :] * stride_k
        mask = mask_n[:, None] & mask_k[None, :]
        
        x = tl.load(input_ptrs, mask=mask, other=float('-inf'))
        curr_max = tl.max(x, axis=1)
        max_val = tl.where(mask_n, tl.maximum(max_val, curr_max), max_val)
    
    for k in range(0, K, TILE_K):
        k_offs = k + tl.arange(0, TILE_K)
        mask_k = k_offs < K
        
        input_ptrs = input_ptr + offs_m_base[:, None] + k_offs[None, :] * stride_k
        mask = mask_n[:, None] & mask_k[None, :]
        
        x = tl.load(input_ptrs, mask=mask, other=0.0)
        x = x - max_val[:, None]
        exp_x = tl.exp(x)
        sum_val += tl.sum(exp_x, axis=1)
    
    for k in range(0, K, TILE_K):
        k_offs = k + tl.arange(0, TILE_K)
        mask_k = k_offs < K
        
        input_ptrs = input_ptr + offs_m_base[:, None] + k_offs[None, :] * stride_k
        mask = mask_n[:, None] & mask_k[None, :]
        
        x = tl.load(input_ptrs, mask=mask, other=0.0)
        x = x - max_val[:, None]
        exp_x = tl.exp(x)
        output = exp_x / sum_val[:, None]
        
        output_ptrs = output_ptr + offs_m_base[:, None] + k_offs[None, :] * stride_k
        tl.store(output_ptrs, output, mask=mask)

@triton.jit
def softmax_kernel_inner(
    output_ptr, input_ptr, M, N, K,
    stride_m, stride_n, stride_k,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr
):
    pid_m = tl.program_id(0)
    
    if pid_m >= M * N:
        return
    
    offs_m = pid_m * stride_m
    max_val = tl.full((1,), float('-inf'), dtype=tl.float32)
    sum_val = tl.zeros((1,), dtype=tl.float32)
    
    for k in range(0, K, TILE_K):
        k_offs = k + tl.arange(0, TILE_K)
        mask = k_offs < K
        
        input_ptrs = input_ptr + offs_m + k_offs * stride_k
        x = tl.load(input_ptrs, mask=mask, other=float('-inf'))
        curr_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, curr_max)
    
    for k in range(0, K, TILE_K):
        k_offs = k + tl.arange(0, TILE_K)
        mask = k_offs < K
        
        input_ptrs = input_ptr + offs_m + k_offs * stride_k
        x = tl.load(input_ptrs, mask=mask, other=0.0)
        x = x - max_val
        exp_x = tl.exp(x)
        sum_val += tl.sum(exp_x, axis=0)
    
    for k in range(0, K, TILE_K):
        k_offs = k + tl.arange(0, TILE_K)
        mask = k_offs < K
        
        input_ptrs = input_ptr + offs_m + k_offs * stride_k
        x = tl.load(input_ptrs, mask=mask, other=0.0)
        x = x - max_val
        exp_x = tl.exp(x)
        output = exp_x / sum_val
        
        output_ptrs = output_ptr + offs_m + k_offs * stride_k
        tl.store(output_ptrs, output, mask=mask)

@triton.jit
def softmax_backward_kernel_non_inner(
    in_grad_ptr, out_grad_ptr, output_ptr, M, N, K,
    stride_m, stride_n, stride_k,
    TILE_K: tl.constexpr, TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    if pid_m >= M:
        return
    
    n_start = pid_n * TILE_N
    n_offs = n_start + tl.arange(0, TILE_N)
    mask_n = n_offs < N
    
    offs_m_base = pid_m * stride_m + n_offs * stride_n
    
    sum_dy = tl.zeros((TILE_N,), dtype=tl.float32)
    
    for k in range(0, K, TILE_K):
        k_offs = k + tl.arange(0, TILE_K)
        mask_k = k_offs < K
        
        grad_ptrs = out_grad_ptr + offs_m_base[:, None] + k_offs[None, :] * stride_k
        output_ptrs = output_ptr + offs_m_base[:, None] + k_offs[None, :] * stride_k
        
        mask = mask_n[:, None] & mask_k[None, :]
        dy = tl.load(grad_ptrs, mask=mask, other=0.0)
        y = tl.load(output_ptrs, mask=mask, other=0.0)
        sum_dy += tl.sum(dy * y, axis=1)
    
    for k in range(0, K, TILE_K):
        k_offs = k + tl.arange(0, TILE_K)
        mask_k = k_offs < K
        
        grad_ptrs = out_grad_ptr + offs_m_base[:, None] + k_offs[None, :] * stride_k
        output_ptrs = output_ptr + offs_m_base[:, None] + k_offs[None, :] * stride_k
        
        mask = mask_n[:, None] & mask_k[None, :]
        dy = tl.load(grad_ptrs, mask=mask, other=0.0)
        y = tl.load(output_ptrs, mask=mask, other=0.0)
        grad = y * (dy - sum_dy[:, None])
        
        in_grad_ptrs = in_grad_ptr + offs_m_base[:, None] + k_offs[None, :] * stride_k
        tl.store(in_grad_ptrs, grad, mask=mask)

@triton.jit
def softmax_backward_kernel_inner(
    in_grad_ptr, out_grad_ptr, output_ptr, M, N, K,
    stride_m, stride_n, stride_k,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr
):
    pid_m = tl.program_id(0)
    
    if pid_m >= M * N:
        return
    
    offs_m = pid_m * stride_m
    sum_dy = tl.zeros((1,), dtype=tl.float32)
    
    for k in range(0, K, TILE_K):
        k_offs = k + tl.arange(0, TILE_K)
        mask = k_offs < K
        
        grad_ptrs = out_grad_ptr + offs_m + k_offs * stride_k
        output_ptrs = output_ptr + offs_m + k_offs * stride_k
        
        dy = tl.load(grad_ptrs, mask=mask, other=0.0)
        y = tl.load(output_ptrs, mask=mask, other=0.0)
        sum_dy += tl.sum(dy * y, axis=0)
    
    for k in range(0, K, TILE_K):
        k_offs = k + tl.arange(0, TILE_K)
        mask = k_offs < K
        
        grad_ptrs = out_grad_ptr + offs_m + k_offs * stride_k
        output_ptrs = output_ptr + offs_m + k_offs * stride_k
        
        dy = tl.load(grad_ptrs, mask=mask, other=0.0)
        y = tl.load(output_ptrs, mask=mask, other=0.0)
        grad = y * (dy - sum_dy)
        
        in_grad_ptrs = in_grad_ptr + offs_m + k_offs * stride_k
        tl.store(in_grad_ptrs, grad, mask=mask)

def heur_tile_k(K):
    if K <= 512:
        return 512
    elif K <= 1024:
        return 1024
    else:
        return 2048

def heur_tile_n_non_inner(N):
    return 64 if N >= 64 else 32

def get_tile_config(dim_size, max_tile, heuristic_div=4):
    tile = triton.next_power_of_2(dim_size)
    tile = min(max(tile // heuristic_div, 16), max_tile)
    return tile

class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        dim = dim if dim >= 0 else x.dim() + dim
        x_shape = x.shape
        x_2d = x.flatten(0, -2) if x.dim() > 2 else x
        M, K = x_2d.shape
        
        is_inner = x.stride()[-1] == 1 if dim == x.dim() -1 else False
        output = torch.empty_like(x_2d)
        
        stride_m = x_2d.stride(0)
        stride_n = 1
        stride_k = x_2d.stride(1) if x_2d.dim() > 1 else 1
        
        if is_inner:
            TILE_K = heur_tile_k(K)
            grid = (M,)
            softmax_kernel_inner[grid](
                output, x_2d, M, 1, K,
                stride_m, stride_n, stride_k,
                TILE_K=TILE_K, ONE_TILE_PER_CTA=True
            )
        else:
            TILE_K = get_tile_config(K, 1024)
            TILE_N = heur_tile_n_non_inner(M)
            grid = (1, triton.cdiv(M, TILE_N))
            softmax_kernel_non_inner[grid](
                output, x_2d, 1, M, K,
                stride_m, stride_n, stride_k,
                TILE_K=TILE_K, TILE_N=TILE_N,
                ONE_TILE_PER_CTA=True
            )
        
        ctx.save_for_backward(output)
        ctx.dim = dim
        return output.reshape(x_shape)

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0]
        dim = ctx.dim
        grad_output_2d = grad_output.flatten(0, -2) if grad_output.dim() > 2 else grad_output
        M, K = grad_output_2d.shape
        
        is_inner = grad_output.stride()[-1] == 1 if dim == grad_output.dim() -1 else False
        in_grad = torch.empty_like(grad_output_2d)
        
        stride_m = grad_output_2d.stride(0)
        stride_n = 1
        stride_k = grad_output_2d.stride(1) if grad_output_2d.dim() > 1 else 1
        
        if is_inner:
            TILE_K = heur_tile_k(K)
            grid = (M,)
            softmax_backward_kernel_inner[grid](
                in_grad, grad_output_2d, output, M, 1, K,
                stride_m, stride_n, stride_k,
                TILE_K=TILE_K, ONE_TILE_PER_CTA=True
            )
        else:
            TILE_K = get_tile_config(K, 1024)
            TILE_N = heur_tile_n_non_inner(M)
            grid = (1, triton.cdiv(M, TILE_N))
            softmax_backward_kernel_non_inner[grid](
                in_grad, grad_output_2d, output, 1, M, K,
                stride_m, stride_n, stride_k,
                TILE_K=TILE_K, TILE_N=TILE_N,
                ONE_TILE_PER_CTA=True
            )
        
        return in_grad.reshape(grad_output.shape), None

def softmax(x, dim=-1):
    return Softmax.apply(x, dim)
