import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def rotary_kernel(
    X,  # Pointer to the input tensor
    COS,  # Pointer to the cosine matrix
    SIN,  # Pointer to the sine matrix
    OUT,  # Pointer to the output tensor
    CU_SEQLENS,  # Pointer to cumulative sequence lengths (optional)
    stride_x_batch,  # Stride of batch dimension in X
    stride_x_head,  # Stride of head dimension in X
    stride_x_seq,  # Stride of sequence dimension in X
    stride_x_dim,  # Stride of feature dimension in X
    stride_cos_seq,  # Stride of sequence dimension in COS
    stride_cos_dim,  # Stride of feature dimension in COS
    stride_sin_seq,  # Stride of sequence dimension in SIN
    stride_sin_dim,  # Stride of feature dimension in SIN
    D,  # Total feature dimension size
    H,  # Number of heads (unused in kernel, for compatibility)
    MAX_SEQ_LEN: tl.constexpr,  # Maximum sequence length (for static shape)
    INTERLEAVED: tl.constexpr,  # Whether data is interleaved
    CONJUGATE: tl.constexpr,  # Whether to apply conjugation
    HAS_CU_SEQLENS: tl.constexpr,  # Whether CU_SEQLENS is provided
    BLOCK_D_PAIRS: tl.constexpr,  # Number of pairs processed per block
):
    # Extract 3D grid coordinates
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Handle variable sequence lengths if CU_SEQLENS is provided
    if HAS_CU_SEQLENS:
        seq_start = tl.load(CU_SEQLENS + pid_batch)
        seq_end = tl.load(CU_SEQLENS + pid_batch + 1)
        current_seq_len = seq_end - seq_start
        if pid_m >= current_seq_len:
            return
        pos = pid_m
    else:
        pos = pid_m

    # Compute base pointers for COS and SIN
    cos_ptr = COS + pos * stride_cos_seq
    sin_ptr = SIN + pos * stride_sin_seq

    # Compute base pointer for the current position in X and OUT
    x_ptr = (
        X
        + pid_batch * stride_x_batch
        + pid_head * stride_x_head
        + pid_m * stride_x_seq
    )
    out_ptr = (
        OUT
        + pid_batch * stride_x_batch
        + pid_head * stride_x_head
        + pid_m * stride_x_seq
    )

    # Iterate over feature dimension in blocks
    d = tl.arange(0, BLOCK_D_PAIRS)
    half_D = D // 2
    mask = d < half_D

    # Determine offsets based on interleaved format
    if INTERLEAVED:
        x0_offsets = 2 * d
        x1_offsets = 2 * d + 1
    else:
        x0_offsets = d
        x1_offsets = d + half_D

    # Load input values
    x0 = tl.load(x_ptr + x0_offsets * stride_x_dim, mask=mask, other=0.0)
    x1 = tl.load(x_ptr + x1_offsets * stride_x_dim, mask=mask, other=0.0)

    # Load cos and sin values
    cos = tl.load(cos_ptr + d * stride_cos_dim, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + d * stride_sin_dim, mask=mask, other=0.0)

    # Apply conjugation if needed
    if CONJUGATE:
        sin = -sin

    # Compute rotated values
    rotated_x0 = x0 * cos - x1 * sin
    rotated_x1 = x0 * sin + x1 * cos

    # Store results
    tl.store(out_ptr + x0_offsets * stride_x_dim, rotated_x0, mask=mask)
    tl.store(out_ptr + x1_offsets * stride_x_dim, rotated_x1, mask=mask)


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    interleaved: bool = False,
    conjugate: bool = False,
    inplace: bool = False,
):
    # Check input dimensions
    assert x.dim() == 4, "Input must be 4-dimensional (batch, seq_len, heads, dim)"
    batch_size, seq_len, num_heads, dim = x.shape
    assert dim % 2 == 0, "Feature dimension must be even for rotary encoding"

    if cu_seqlens is not None:
        assert cu_seqlens.is_cuda, "cu_seqlens must be on CUDA device"
        assert cu_seqlens.dim() == 1, "cu_seqlens must be 1-dimensional"
        assert cu_seqlens.size(0) == batch_size + 1, "cu_seqlens must have batch_size + 1 elements"
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    else:
        max_seqlen = seq_len

    # Initialize output tensor
    out = x if inplace else torch.empty_like(x)

    # Setup kernel grid dimensions
    grid = (batch_size, num_heads, max_seqlen)

    # Tensor strides for X
    stride_x_batch = x.stride(0)
    stride_x_seq = x.stride(1)
    stride_x_head = x.stride(2)
    stride_x_dim = x.stride(3)

    # Check cos/sin dimensions
    assert cos.shape == sin.shape, "COS and SIN must have the same shape"
    assert cos.dim() == 2, "COS and SIN must be 2-dimensional (seq_len, dim//2)"
    cos_seq_len, cos_dim = cos.shape
    assert cos_dim == dim // 2, "COS/SIN dim must be half of input feature dim"

    # Tensor strides for COS and SIN
    stride_cos_seq = cos.stride(0)
    stride_cos_dim = cos.stride(1)
    stride_sin_seq = sin.stride(0)
    stride_sin_dim = sin.stride(1)

    # Tuning parameter: number of pairs processed per block (adjust based on hardware)
    BLOCK_D_PAIRS = 128  # Each block processes 128 pairs (256 elements for interleaved)

    # Launch kernel
    rotary_kernel[grid](
        x, cos, sin, out, cu_seqlens,
        stride_x_batch, stride_x_head, stride_x_seq, stride_x_dim,
        stride_cos_seq, stride_cos_dim,
        stride_sin_seq, stride_sin_dim,
        dim, num_heads, max_seqlen,
        interleaved, conjugate, cu_seqlens is not None,
        BLOCK_D_PAIRS,
    )

    return out
