import torch
import triton
import triton.language as tl
from typing import Optional

# Triton kernel for forward pass
@triton.jit
def _normalize_pw_dist_fwd(
    x1_ptr, 
    x2_ptr, 
    out_dist_ptr, 
    out_norm_ptr,
    n_elements_x1,
    n_elements_out,
    stride_x1_batch,
    stride_x1_vec,
    stride_out_dist_batch,
    stride_out_dist_vec,
    stride_out_norm_batch,
    stride_out_norm_vec,
    M: tl.constexpr,
    N: tl.constexpr,
    VEC_LENGTH: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    Abatch_idx = pid_m // N
    Ablock_idx = pid_m % N
    
    x1_ptrs = x1_ptr + (
            Abatch_idx * stride_x1_batch +
            Ablock_idx * stride_x1_vec
            )
    
    out_dist_ptrs = out_dist_ptr + (
                    Abatch_idx * stride_out_dist_batch +
                    Ablock_idx * stride_out_dist_vec
                   ) 
    
    out_norm_ptrs = out_norm_ptr + (
                    Abatch_idx * stride_out_norm_batch +
                    Ablock_idx * stride_out_norm_vec
                    )
    
    mask_m = (M - (pid_m+1)*BLOCK_SIZE_M) >= 0
    mask_n = (N - BLOCK_SIZE_N) >= 0
    
    x1_vec = tl.load(x1_ptrs + rn, mask=((BLOCK_SIZE_M * vec_length) - (pid_m+1)*BLOCK_SIZE_M*rn) >= 0, padding_option='zero')
    x2_vec = tl.load(x2_ptrs + rn, mask=(BLOCK_SIZE_M - rn) >= 0, padding_option='zero')

    diff = tl.abs(x1_vec - tl.trans(x2_vec))  
    dist = tl.max(diff, axis=0)
    
    inv_denom = tl.math.rsqrt(dist + eps_distance)
    norm = tl.sum(inv_denom, axis=0)

    tl.store(out_dist_ptrs + rn, dist, mask=mask_n)
    tl.store(out_norm_ptrs + rn, norm, mask=mask_n)

# Triton kernel for backward pass
@triton.jit
def _normalize_pw_dist_bwd(
    grad_out_dist_ptr,
    grad_out_norm_ptr,
    x1_ptr,
    x2_ptr,
    grad_x1_ptr,
    grad_x2_ptr,
    n_elements_x1,
    n_elements_out,
    stride_x1_batch,
    stride_x1_vec,
    stride_out_grad_batch,
    stride_out_grad_vec,
    stride_x_grad_batch,
    stride_x_grad_vec,
    M: tl.constexpr,
    N: tl.constexpr,
    VEC_LENGTH: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    Abatch_idx = pid_m // N
    Ablock_idx = pid_m % N
    
    x1_ptrs = x1_ptr + (
            Abatch_idx * stride_x1_batch +
            Ablock_idx * stride_x1_vec
            )
    
    grad_out_ptrs = grad_out_dist_ptr + (
                    Abatch_idx * stride_out_grad_batch +
                    Ablock_idx * stride_out_grad_vec
                   ) 
    
    grad_x1_ptrs = grad_x1_ptr + (
                    Abatch_idx * stride_x_grad_batch +
                    Ablock_idx * stride_x_grad_vec
                  )
    
    mask_m = (M - (pid_m+1)*BLOCK_SIZE_M) >= 0
    mask_n = (N - BLOCK_SIZE_N) >= 0
    
    x1_vec = tl.load(x1_ptrs + rn, mask=((BLOCK_SIZE_M * vec_length) - (pid_m+1)*BLOCK_SIZE_M*rn) >= 0, padding_option='zero')
    
    denom = tl.load(grad_out_ptrs + rn, mask=mask_n,)
    inv_denom = tl.math.rsqrt(denom + eps_distance)
    grad_norm = tl.load(grad_out_norm_ptr + rn, mask=mask_n,) * inv_denom
    
    grad_x1 = (inv_denom * (tl.sum((tl.abs(x1_vec - tl.trans(x2_vec)) * grad_norm), axis=0)))

    tl.store(grad_x1_ptrs + rn, grad_x1, mask=mask_m)

# Wrapper function for forward pass
def normalize_pw_dist_fwd(x1: torch.Tensor, x2: torch.Tensor, p_distance: float, eps_distance: float, keepdim: bool, p_norm: float, dim_norm: int, eps_norm: float):
    if p_distance != 2.0 or eps_distance != 1e-6:
        raise NotImplementedError("For now, only L2 distance with eps=1e-6 is allowed.")
    if not keepdim:
        x1 = torch.squeeze(x1, dim_norm)
        x2 = torch.squeeze(x2, dim_norm)
        
    M, N, vec_length = x1.size(0), x1.size(-1), x2.size(-1)
    
    out_shape = list(x1.shape)[:-1] + [N] if keepdim else list(x1.shape)[:-1]
    out_dist = torch.zeros(out_shape, dtype=x1.dtype, device=x1.device)
    out_norm = torch.zeros(out_shape, dtype=x1.dtype, device=x1.device)
    
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    if vec_length <= 64:
        BLOCK_SIZE_M = 256
    elif vec_length <= 32:
        BLOCK_SIZE_M = 512
        
    if vec_length > 128:
        BLOCK_SIZE_N = 64
    elif vec_length > 256:
        BLOCK_SIZE_N = 32
        
    if vec_length == 1:
        out_dist.copy_(out_norm)
        return out_dist, out_norm
    
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"])
    )
    
    with torch.cuda.device(x1.device.index):
        _normalize_pw_dist_fwd[grid](
            x1,
            x2,
            out_dist,
            out_norm,
            M, N,
            x1.stride(0),
            x1.stride(-1),
            out_dist.stride(0),
            out_dist.stride(-1),
            out_norm.stride(0),
            out_norm.stride(-1),
            M, N,
            vec_length,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )
    return out_dist, out_norm

# Wrapper function for backward pass
def normalize_pw_dist_bwd(
    grad_out: torch.Tensor, 
    x1: torch.Tensor, 
    x2: torch.Tensor, 
    save_x1: Optional[torch.Tensor],
    save_x2: Optional[torch.Tensor],
    p_distance: float,
    eps_distance: float,
    keepdim: bool,
    p_norm: float,
    dim_norm: int,
    eps_norm: float,
):
    if p_distance != 2.0 or eps_distance != 1e-6:
        raise NotImplementedError("For now, only L2 distance with eps=1e-6 is allowed.")
    if not keepdim:
        x1 = torch.squeeze(x1, dim_norm)
        x2 = torch.squeeze(x2, dim_norm)
    
    M, N, vec_length = x1.size(0), x1.size(-1), x2.size(-1)
    
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    if vec_length <= 64:
        BLOCK_SIZE_M = 256
    elif vec_length <= 32:
        BLOCK_SIZE_M = 512
        
    if vec_length > 128:
        BLOCK_SIZE_N = 64
    elif vec_length > 256:
        BLOCK_SIZE_N = 32
        
    if vec_length == 1:
        grad_x1 = grad_out.clone().detach()
        grad_x2 = -grad_x1
        return grad_x1.to(x1), grad_x2.to(x2)
    
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    
    grad_x1 = torch.zeros_like(x1, dtype=x1.dtype)
    grad_x2 = torch.zeros_like(x2, dtype=x2.dtype)
    
    with torch.cuda.device(x1.device.index):
        _normalize_pw_dist_bwd[grid](
            grad_out,
            save_x,
            x1,
            x2,
            grad_x1,
            grad_x2,
            M, N,
            x1.stride(0),
            x1.stride(-1),
            grad_out.stride(0),
            grad_out.stride(-1),
            grad_x1.stride(0),
            grad_x1.stride(-1),
            M, N,
            vec_length,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )
    return grad_x1.to(x1), grad_x2.to(x2)
