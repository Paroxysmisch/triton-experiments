import torch
import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    X,
    COS,
    SIN,
    OUT,
    stride_x_batch,
    stride_x_head,
    stride_x_seq,
    stride_x_head_dim,
    stride_out_batch,
    stride_out_head,
    stride_out_seq,
    stride_out_head_dim,
    seq_lens,
    IS_VARLEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    if IS_VARLEN:
        cur_seq_len = tl.load(seq_lens + pid_m)
    else:
        cur_seq_len = tl.num_programs(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_x = (
        offs_m[:, None] * stride_x_batch
        + (offs_n[None, :] // 2) * stride_x_head
        + (offs_n[None, :] % 2) * stride_x_seq
        + offs_n[None, :] * stride_x_head_dim
    )

    if INTERLEAVED:
        x = tl.load(X + offs_x, mask=(offs_n[None, :] < cur_seq_len), other=0.0)
    else:
        offs_x_odd = offs_x + stride_x_head_dim // 2
        x_even = tl.load(
            X + offs_x, mask=((offs_n[None, :]) < cur_seq_len) & (offs_n[None, :] % 2 == 0), other=0.0
        )
        x_odd = tl.load(
            X + offs_x_odd, mask=((offs_n[None, :]) < cur_seq_len) & (offs_n[None, :] % 2 == 1), other=0.0
        )
        x = tl.concatenate([x_even, x_odd], axis=1)

    if INTERLEAVED:
        offs_out = offs_m[:, None] * stride_out_batch + offs_n[None, :] * stride_out_head_dim
    else:
        offs_out = (
            offs_m[:, None] * stride_out_batch
            + (offs_n[None, :] // 2) * stride_out_head
            + (offs_n[None, :] % 2) * stride_out_seq
            + offs_n[None, :] * stride_out_head_dim
        )

    cos = tl.load(COS + offs_x, mask=(offs_n[None, :] < cur_seq_len), other=0.0)
    if CONJUGATE:
        sin = tl.load(SIN + offs_x, mask=(offs_n[None, :] < cur_seq_len), other=0.0)
        sin = -sin
    else:
        sin = tl.load(SIN + offs_x, mask=(offs_n[None, :] < cur_seq_len), other=0.0)
    x_real = x[None, :] * cos
    x_imag = x[None, :] * sin
    out = x_real.to(x.dtype)
    tl.store(OUT + offs_out, out, mask=(offs_n[None, :] < cur_seq_len))


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: torch.Tensor,
    seq_lens: torch.Tensor = None,
    is_varlen: bool = False,
    dim=-2,
    interleaved: bool = True,
    conjugate: bool = False,
    config=None,
):
    assert x.dtype == cos.dtype == sin.dtype == out.dtype
    assert x.is_contiguous() and cos.is_contiguous() and sin.is_contiguous() and out.is_contiguous()
    assert x.shape[dim] == cos.shape[0] and x.shape[dim] == sin.shape[0]
    assert cos.shape[0] // 2 == sin.shape[0] // 2

    if seq_lens is None:
        seq_lens = torch.full((out.shape[1],), out.shape[2], dtype=torch.int32, device=cos.device)
    else:
        assert seq_lens.shape[0] == out.shape[1]

    if config is None:
        config = {"num_warps": 1, "num_stages": 2}

    assert x.shape[dim] % 2 == 0

    batch, seq, head, head_dim = (
        out.shape[0],
        out.shape[2],
        out.shape[1],
        out.shape[3],
    )

    if dim == -2:
        stride_x_batch = 1
        stride_x_head = x.stride(0) * x.shape[0] // batch
        stride_x_seq = x.stride(1)
        stride_x_head_dim = x.stride(2)
    else:
        stride_x_batch = x.stride(0)
        stride_x_head = x.stride(1)
        stride_x_seq = x.stride(2)
        stride_x_head_dim = x.stride(3)

    if dim == -2:
        stride_out_batch = 1
        stride_out_head = out.stride(0) * out.shape[0] // batch
        stride_out_seq = out.stride(1)
        stride_out_head_dim = out.stride(2)
    else:
        stride_out_batch = out.stride(0)
        stride_out_head = out.stride(1)
        stride_out_seq = out.stride(2)
        stride_out_head_dim = out.stride(3)

    if is_varlen:
        assert seq_lens.stride(0) == 1

    BLOCK_M = triton.next_power_of_2(head)
    BLOCK_K = triton.next_power_of_2(head_dim)

    grid = lambda meta: (triton.cdiv(head, meta["BLOCK_M"]), triton.cdiv(head_dim, meta["BLOCK_K"]))

    rotary_kernel[grid](
        x,
        cos,
        sin,
        out,
        stride_x_batch,
        stride_x_head,
        stride_x_seq,
        stride_x_head_dim,
        stride_out_batch,
        stride_out_head,
        stride_out_seq,
        stride_out_head_dim,
        seq_lens,
        is_varlen,
        BLOCK_M,
        BLOCK_K,
        interleaved,
        conjugate,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
