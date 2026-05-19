import torch
import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,
    stride_x_batch,
    stride_x_seq,
    stride_x_head,
    stride_x_headdim,
    stride_cos_seq,
    stride_cos_headdim,
    stride_sin_seq,
    stride_sin_headdim,
    stride_output_batch,
    stride_output_seq,
    stride_output_head,
    stride_output_headdim,
    headdim,
    max_seqlen,
    IS_VARLEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_seq = tl.program_id(2)

    seq_offsets = pid_seq * BLOCK_M + tl.arange(0, BLOCK_M)
    seq_mask = seq_offsets < max_seqlen

    feature_offsets = tl.arange(0, BLOCK_K)
    headdim_half = headdim // 2
    feature_mask = feature_offsets < headdim_half

    mask = seq_mask[:, None] & feature_mask[None, :]

    x_batch_head_ptr = (
        x_ptr
        + pid_batch * stride_x_batch
        + pid_head * stride_x_head
    )

    cos_sin_offsets = seq_offsets[:, None] * stride_cos_seq + feature_offsets[None, :] * stride_cos_headdim
    cos = tl.load(cos_ptr + cos_sin_offsets, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + cos_sin_offsets, mask=mask, other=0.0)

    if CONJUGATE:
        sin = -sin

    if INTERLEAVED:
        feature_idx0 = 2 * feature_offsets
        feature_idx1 = 2 * feature_offsets + 1
    else:
        feature_idx0 = feature_offsets
        feature_idx1 = feature_offsets + headdim_half

    valid0 = feature_idx0 < headdim
    valid1 = feature_idx1 < headdim
    feature_pair_mask = valid0 & valid1

    mask_pair = seq_mask[:, None] & feature_pair_mask[None, :]

    x0_ptrs = (
        x_batch_head_ptr
        + seq_offsets[:, None] * stride_x_seq
        + feature_idx0[None, :] * stride_x_headdim
    )
    x1_ptrs = (
        x_batch_head_ptr
        + seq_offsets[:, None] * stride_x_seq
        + feature_idx1[None, :] * stride_x_headdim
    )

    x0 = tl.load(x0_ptrs, mask=mask_pair, other=0.0)
    x1 = tl.load(x1_ptrs, mask=mask_pair, other=0.0)

    new_x0 = x0 * cos - x1 * sin
    new_x1 = x0 * sin + x1 * cos

    output_batch_head_ptr = (
        output_ptr
        + pid_batch * stride_output_batch
        + pid_head * stride_output_head
    )

    output0_ptrs = (
        output_batch_head_ptr
        + seq_offsets[:, None] * stride_output_seq
        + feature_idx0[None, :] * stride_output_headdim
    )
    output1_ptrs = (
        output_batch_head_ptr
        + seq_offsets[:, None] * stride_output_seq
        + feature_idx1[None, :] * stride_output_headdim
    )

    tl.store(output0_ptrs, new_x0, mask=mask_pair)
    tl.store(output1_ptrs, new_x1, mask=mask_pair)

def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    conjugate: bool = False,
    inplace: bool = False,
    is_varlen: bool = False,
    BLOCK_M: int = 64,
    BLOCK_K: int = 64,
):
    assert x.dtype in [torch.float16, torch.float32], "Unsupported input dtype"
    assert cos.dtype == x.dtype and sin.dtype == x.dtype, "Cos/Sin dtype mismatch"
    assert x.ndim == 4, "Input must be 4D (batch, seq, num_heads, headdim)"
    batch_size, seq_len, num_heads, headdim = x.shape
    assert headdim % 2 == 0, "Headdim must be even"

    if not x.is_contiguous():
        x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    output = x if inplace else torch.empty_like(x)

    grid = (batch_size, num_heads, (seq_len + BLOCK_M - 1) // BLOCK_M)

    stride_x_batch, stride_x_seq, stride_x_head, stride_x_headdim = x.stride()
    stride_output_batch, stride_output_seq, stride_output_head, stride_output_headdim = output.stride()

    stride_cos_seq, stride_cos_headdim = cos.stride()[0], cos.stride()[1] if cos.ndim > 1 else 0
    stride_sin_seq, stride_sin_headdim = sin.stride()[0], sin.stride()[1] if sin.ndim > 1 else 0

    rotary_kernel[grid](
        x, cos, sin, output,
        stride_x_batch, stride_x_seq, stride_x_head, stride_x_headdim,
        stride_cos_seq, stride_cos_headdim,
        stride_sin_seq, stride_sin_headdim,
        stride_output_batch, stride_output_seq, stride_output_head, stride_output_headdim,
        headdim, seq_len,
        IS_VARLEN=is_varlen,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        INTERLEAVED=interleaved,
        CONJUGATE=conjugate,
    )

    return output
