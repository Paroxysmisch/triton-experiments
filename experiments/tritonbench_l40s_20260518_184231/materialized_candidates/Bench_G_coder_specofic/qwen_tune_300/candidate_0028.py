import torch
import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    X,
    COS,
    SIN,
    batch,
    seq_len,
    nheads,
    headdim,
    stride_x_batch,
    stride_x_seq,
    stride_x_head,
    stride_x_headdim,
    stride_cos_batch,
    stride_cos_head,
    stride_cos_headdim,
    stride_cos_seq,
    stride_sin_batch,
    stride_sin_head,
    stride_sin_headdim,
    stride_sin_seq,
    IS_VARLEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    X += pid_batch * stride_x_batch + pid_head * stride_x_head
    COS += pid_batch * stride_cos_batch + pid_head * stride_cos_head
    SIN += pid_batch * stride_sin_batch + pid_head * stride_sin_head

    if not IS_VARLEN:
        seq_index = tl.arange(0, BLOCK_M)
        headdim_index = tl.arange(0, BLOCK_K)
        X += seq_index[:, None] * stride_x_seq + headdim_index[None, :] * stride_x_headdim
        COS += seq_index * stride_cos_seq + headdim_index * stride_cos_headdim
        SIN += seq_index * stride_sin_seq + headdim_index * stride_sin_headdim
    else:
        pid_seq = tl.program_id(axis=3)
        X += pid_seq * stride_x_seq
        COS += pid_seq * stride_cos_seq
        SIN += pid_seq * stride_sin_seq
        seq_index = pid_seq * BLOCK_M + tl.arange(0, BLOCK_M)
        headdim_index = tl.arange(0, BLOCK_K)

    dim_index = pid_head * BLOCK_K + tl.arange(0, BLOCK_K)
    dim_mask = dim_index < headdim
    X += dim_index[None, :] * stride_x_headdim
    COS += dim_index * stride_cos_headdim
    SIN += dim_index * stride_sin_headdim

    cos = tl.load(COS + seq_index * stride_cos_seq + dim_index * stride_cos_headdim, mask=dim_mask, other=0.0)
    sin = tl.load(SIN + seq_index * stride_sin_seq + dim_index * stride_sin_headdim, mask=dim_mask, other=0.0)
    if CONJUGATE:
        sin = -sin
    x0 = tl.load(X + 0 + stride_x_headdim // 2, mask=dim_mask, other=0.0)
    x1 = tl.load(X + 1 + stride_x_headdim // 2, mask=dim_mask, other=0.0)
    x0_out = x0 * cos - x1 * sin
    x1_out = x1 * cos + x0 * sin
    if not INTERLEAVED:
        tl.store(X + 0 + stride_x_headdim // 2, x0_out, mask=dim_mask)
        tl.store(X + 1 + stride_x_headdim // 2, x1_out, mask=dim_mask)
    else:
        tl.store(X + 0, x0_out, mask=dim_mask)
        tl.store(X + 1, x1_out, mask=dim_mask)


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    varlen: bool = False,
    inplace: bool = False,
    interleaved: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    """Applies Rotary Position Embedding on the input tensor."""
    assert x.dim() == 4 and cos.dim() == 3 and sin.dim() == 3
    assert x.size(0) == cos.size(0) and x.size(1) == cos.size(1) and x.size(2) == cos.size(2)
    assert x.size(0) == sin.size(0) and x.size(1) == sin.size(1) and x.size(2) == sin.size(2)
    assert x.is_contiguous()
    if not inplace:
        x = x.clone()
    batch, seq_len, nheads, headdim = x.shape
    grid = (triton.cdiv(seq_len, 32) if not varlen else (seq_len,), batch, nheads)
    kwargs = {
        "varlen": varlen,
        "interleaved": interleaved,
        "conjugate": conjugate,
    }
    BLOCK_M = 32 if not varlen else 1
    BLOCK = 16 if headdim <= 128 else 8
    rotary_kernel[grid](
        x,
        cos,
        sin,
        batch,
        seq_len,
        nheads,
        headdim,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        cos.stride(3),
        sin.stride(0),
        sin.stride(1),
        sin.stride(2),
        sin.stride(3),
        IS_VARLEN=varlen,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK,
        num_warps=4 if BLOCK_M == 32 else 1,
        num_stages=2,
        **kwargs,
    )
    return x
