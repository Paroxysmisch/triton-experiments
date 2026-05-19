import torch
import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    X_ptr, COS_ptr, SIN_ptr, OUT_ptr,
    stride_xb, stride_xh, stride_xm, stride_xk,
    stride_cb, stride_ch, stride_cm,
    stride_sb, stride_sh, stride_sm,
    stride_ob, stride_oh, stride_om, stride_ok,
    seq_len, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    IS_VARLEN: tl.constexpr, INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr
):
    pid = tl.program_id(0)
    num_heads = tl.num_programs(0) // seq_len
    batch_id = pid // (num_heads * seq_len)
    head_id = (pid % (num_heads * seq_len)) // seq_len
    seq_id = pid % seq_len

    X_ptr += batch_id * stride_xb + head_id * stride_xh + seq_id * stride_xm
    COS_ptr += batch_id * stride_cb + head_id * stride_ch + seq_id * stride_cm
    SIN_ptr += batch_id * stride_sb + head_id * stride_sh + seq_id * stride_sm
    OUT_ptr += batch_id * stride_ob + head_id * stride_oh + seq_id * stride_om

    if IS_VARLEN:
        seq_len = tl.load(seq_len + batch_id)

    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < head_dim

    for k in range(0, head_dim, BLOCK_K):
        x_ptr = X_ptr + k * (1 if INTERLEAVED else stride_xk)
        out_ptr = OUT_ptr + k * stride_ok

        x = tl.load(x_ptr + offs_k * (stride_xk if INTERLEAVED else 1), mask=mask, other=0.0)
        cos = tl.load(COS_ptr + offs_k, mask=mask, other=1.0)
        sin = tl.load(SIN_ptr + offs_k, mask=mask, other=0.0)

        if INTERLEAVED:
            x_real, x_imag = x[0::2], x[1::2]
        else:
            x_real, x_imag = x[:BLOCK_K//2], x[BLOCK_K//2:]

        if CONJUGATE:
            sin = -sin

        out_real = x_real * cos - x_imag * sin
        out_imag = x_real * sin + x_imag * cos

        if INTERLEAVED:
            out = tl.where(tl.arange(0, BLOCK_K) % 2 == 0, out_real, out_imag)
        else:
            out = tl.concat(out_real, out_imag)

        tl.store(out_ptr + offs_k * stride_ok, out, mask=mask)

def apply_rotary(x, cos, sin, seq_lens=None, interleaved=False, inplace=False, conjugate=False):
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], f"Unsupported dtype: {x.dtype}"
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert cos.is_contiguous(), "Cosine tensor must be contiguous"
    assert sin.is_contiguous(), "Sine tensor must be contiguous"

    batch, seq_len, n_heads, head_dim = x.shape
    assert cos.shape == sin.shape == (seq_len, head_dim), "Invalid shapes for cos/sin tensors"

    if seq_lens is not None:
        assert seq_lens.shape == (batch,), "seq_lens must have shape (batch,)"
        max_seq_len = seq_lens.max().item()
        assert max_seq_len <= seq_len, "max_seq_len must be <= seq_len"
        is_varlen = True
    else:
        is_varlen = False
        max_seq_len = seq_len

    if inplace:
        out = x
    else:
        out = torch.empty_like(x)

    BLOCK_M = 1
    BLOCK_K = 32

    def grid(meta):
        return (batch * n_heads * max_seq_len,)

    rotary_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(2), x.stride(1), x.stride(3),
        cos.stride(0), 0, 0,
        sin.stride(0), 0, 0,
        out.stride(0), out.stride(2), out.stride(1), out.stride(3),
        seq_lens if is_varlen else None, head_dim,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        IS_VARLEN=is_varlen, INTERLEAVED=interleaved,
        CONJUGATE=conjugate,
        num_warps=1
    )

    return out
