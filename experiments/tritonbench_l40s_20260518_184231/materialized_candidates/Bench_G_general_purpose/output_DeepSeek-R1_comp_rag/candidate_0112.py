import torch
import triton
import triton.language as tl
from typing import Optional, Union

@triton.jit
def rotary_kernel(
    OUT, X, COS, SIN, CU_SEQLENS, SEQLEN_OFFSETS,
    seqlen, nheads, rotary_dim, seqlen_ro, CACHE_KEY_SEQLEN,
    stride_out_batch, stride_out_seqlen, stride_out_nheads, stride_out_headdim,
    stride_x_batch, stride_x_seqlen, stride_x_nheads, stride_x_headdim,
    BLOCK_K: tl.constexpr, IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr, INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr, BLOCK_M: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)
    rotary_dim_half = rotary_dim // 2

    if IS_VARLEN:
        seq_start = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - seq_start
        X += seq_start * stride_x_seqlen + pid_head * stride_x_nheads
        OUT += seq_start * stride_out_seqlen + pid_head * stride_out_nheads
    else:
        X += pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT += pid_batch * stride_out_batch + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen:
        return
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rm_cs = rm + (tl.load(SEQLEN_OFFSETS + pid_batch) if IS_SEQLEN_OFFSETS_TENSOR else SEQLEN_OFFSETS)
    rk = tl.arange(0, BLOCK_K)
    rk_half = rk[:BLOCK_K//2]

    if not INTERLEAVED:
        x_ptr = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
        cos_ptr = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        sin_ptr = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        
        cos = tl.load(cos_ptr, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=1.0)
        sin = tl.load(sin_ptr, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=0.0)
        
        x0 = tl.load(x_ptr, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0)
        x1 = tl.load(x_ptr + rotary_dim_half * stride_x_headdim, 
                    mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0)

        if CONJUGATE: sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos

        out_ptr = OUT + rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim
        tl.store(out_ptr, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
        tl.store(out_ptr + rotary_dim_half * stride_out_headdim, o1, 
                mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
    else:
        rk_swap = rk + ((rk + 1) % 2) * 2 - 1  # Pairwise swapping
        rk_repeat = rk // 2
        
        cos_ptr = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        sin_ptr = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        cos = tl.load(cos_ptr, mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half), other=1.0)
        sin = tl.load(sin_ptr, mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half), other=0.0)
        
        x0_ptr = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
        x1_ptr = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)
        x0 = tl.load(x0_ptr, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0)
        x1 = tl.load(x1_ptr, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0)

        if CONJUGATE: sin = -sin
