import torch
import triton
import triton.language as tl
from typing import Optional, Union

@triton.jit
def rotary_kernel(
    # Tensor pointers
    OUT, X, COS, SIN, CU_SEQLENS, SEQLEN_OFFSETS,
    # Dimensions
    seqlen, nheads, rotary_dim, seqlen_ro, CACHE_KEY_SEQLEN,
    # Strides
    stride_out_batch, stride_out_seqlen, stride_out_nheads, stride_out_headdim,
    stride_x_batch, stride_x_seqlen, stride_x_nheads, stride_x_headdim,
    # Meta-parameters
    BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Program ID management
    pid_m = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)
    rotary_dim_half = rotary_dim // 2

    # Handle variable length sequences
    if IS_VARLEN:
        seq_start = tl.load(CU_SEQLENS + pid_batch)
        seq_end = tl.load(CU_SEQLENS + pid_batch + 1)
        seqlen = seq_end - seq_start
        X += seq_start * stride_x_seqlen + pid_head * stride_x_nheads
        OUT += seq_start * stride_out_seqlen + pid_head * stride_out_nheads
    else:
        X += pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT += pid_batch * stride_out_batch + pid_head * stride_out_nheads

    # Early exit if block exceeds sequence length
    if pid_m * BLOCK_M >= seqlen:
        return

    # Position indices calculations
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    else:
        rm_cs = rm + SEQLEN_OFFSETS

    # Main computation logic
    if not INTERLEAVED:
        # Non-interleaved memory layout
        rk_half = tl.arange(0, BLOCK_K // 2)
        X_ptr = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
        COS_ptr = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        SIN_ptr = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])

        # Load data with masking
        cos = tl.load(COS_ptr, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=1.0)
        sin = tl.load(SIN_ptr, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=0.0)
        x0 = tl.load(X_ptr, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0)
        x1 = tl.load(X_ptr + rotary_dim_half * stride_x_headdim, 
                    mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0)

        # Apply rotation
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos

        # Store results
        OUT_ptr = OUT + (rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim)
        tl.store(OUT_ptr, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
        tl.store(OUT_ptr + rotary_dim_half * stride_out_headdim, o1, 
                mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
    else:
        # Interleaved memory layout
        rk = tl.arange(0, BLOCK_K)
        rk_swap = rk + ((rk + 1) % 2) * 2 - 1  # Swaps even and odd indices
        rk_repeat = rk // 2

        # Load data with interleaved pattern
        X0_ptr = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
        X1_ptr = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)
        COS_ptr = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        SIN_ptr = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])

        cos = tl.load(COS_ptr, mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half), other=1.0)
        sin = tl.load(SIN_ptr, mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half), other=0.0)
        x0 = tl.load(X0_ptr, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0)
        x1 = tl.load(X1_ptr, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0)

        # Apply rotation with conjugation
        if CONJUGATE:
            sin = -sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        
        # Combine results based on even/odd indices
        out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
        OUT_ptr = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim)
        tl.store(OUT_ptr, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))

def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    # Input validation and shape management
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, "max_seqlen required for variable length"
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.size(0) - 1
        seqlen = max_seqlen

    # Rotary dimension checks
    seqlen_ro, rotary_dim = cos.shape
    rotary_dim *= 2
    assert sin.shape == cos.shape, "Cos and Sin must have same shape"
    assert rotary_dim <= headdim, "Rotary dimension exceeds head dimension"
    assert seqlen_ro >= seqlen, "Rotary sequence length insufficient"

    # Tensor preparation
    cos, sin = cos.contiguous(), sin.contiguous()
    output = x if inplace else torch.empty_like(x)
    if not inplace and rotary_dim < headdim:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    # Block size configuration
    BLOCK_K = min(triton.next_power_of_2(rotary_dim), 256)
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)
    grid = (triton.cdiv(seqlen, BLOCK_M), batch, nheads)

    # Kernel launch
    rotary_kernel[grid](
        output, x, cos, sin, cu_seqlens, seqlen_offsets,
        seqlen, nheads, rotary_dim, seqlen_ro, seqlen // 128,
        output.stride(0) if not is_varlen else 0,
        output.stride(-3), output.stride(-2), output.stride(-1),
        x.stride(0) if not is_varlen else 0,
        x.stride(-3), x.stride(-2), x.stride(-1),
        BLOCK_K, isinstance(seqlen_offsets, torch.Tensor),
        is_varlen, interleaved, conjugate, BLOCK_M
    )
    return output

# Example usage
x = torch.randn(2, 1024, 16, 128, device='cuda', dtype=torch.float16)
cos = torch.randn(1024, 32, device='cuda', dtype=torch.float16)
sin = torch.randn(1024, 32, device='cuda', dtype=torch.float16)

# Apply rotary positional encoding
output = apply_rotary(
    x, cos, sin,
    seqlen_offsets=0,
    interleaved=False,
    inplace=False
)
