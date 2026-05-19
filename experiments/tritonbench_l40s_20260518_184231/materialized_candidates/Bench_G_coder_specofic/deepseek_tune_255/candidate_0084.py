import torch
import triton
import triton.language as tl
from typing import Optional
from .triton_utils.kernels import BLOCK_M, BLOCK_N, BLOCK_DMODEL, get_config

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, sm_scale, Out,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vk, stride_vn,
    stride_b0h, stride_b0m, stride_b0k,
    stride_oh, stride_om, stride_on,
    BIAS_LAST_SIZE,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + off_h * stride_qh + off_b * stride_qm * stride_qk + (offs_m[:, None] * stride_qk + offs_d[None, :])
    k_ptrs = K + off_h * stride_kh + off_b * stride_kn * stride_kk + (offs_n[:, None] * stride_kk + offs_d[None, :])
    v_ptrs = V + off_h * stride_vh + off_b * stride_vk * stride_vn + (offs_n[:, None] * stride_vk + offs_d[None, :])
    b0_ptrs = B0 + off_h * stride_b0h + off_b * stride_b0m * stride_b0k + (offs_m[:, None] * stride_b0k + offs_n[None, :])
    t_ptrs = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    q = tl.load(q_ptrs)
    if IS_CAUSAL:
        causal_mask = (offs_m[:, None] >= (offs_m[None, :] + 1))
    for start_n in range(0, offs_n.shape[0], BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + start_n * stride_kn)
        if IS_CAUSAL:
            causal_mask = causal_mask and (offs_m[:, None] >= (start_n + offs_n[None, :] + 1))
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        if BIAS_LAST_SIZE > 0:
            bias = tl.load(b0_ptrs + start_n * stride_b0k)
            qk = qk + bias
        t_ptrs = t_ptrs + qk
        if IS_CAUSAL:
            t_ptrs = t_ptrs * (1 - causal_mask)
        else:
            t_ptrs = t_ptrs
        t = tl.math.exp2(sm_scale * t_ptrs)
        tl.store(t_ptrs, 0)
        p = tl.math.exp2(sm_scale * t)
        v = tl.load(v_ptrs + start_n * stride_vk)
        acc = acc * (p[:, None])
        acc = tl.dot(t, v, acc)
        if BIAS_LAST_SIZE > 0:
            bias = tl.load(b0_ptrs + start_n * stride_b0k)
            acc = acc + bias * p[:, None]
    acc = acc.to(OUT_DTYPE)
    o_ptrs = Out + off_h * stride_oh + off_b * stride_om * stride_on + (offs_m[:, None] * stride_on + offs_d[None, :])
    tl.store(o_ptrs, acc)


def _attention_rel_h_rel_w_kernel_aligned_device(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    rel_h_w: torch.Tensor,
    sm_scale: float,
    out: Optional[torch.Tensor],
    bias: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    tuning_opt: dict = {},
):
    B, H, N, D = q.shape
    assert k.shape == (B, H, N, D)
    assert v.shape == (B, H, N, D)
    assert q.shape == out.shape
    if bias is not None:
        assert bias.shape == (B, H, N, N) or bias.shape == (B, H, N)
    if rel_h_w.ndim == 4:
        assert rel_h_w.shape == (B, H, N, N)
        BIAS_LAST_SIZE = N * N
    else:
        assert rel_h_w.shape == (B, H, N)
        BIAS_LAST_SIZE = N
    assert sm_scale > 0
    OUT_DTYPE = q.dtype
    if out.dtype.is_fp16:
        if torch.cuda.device(q.device).get_capability() >= (7, 0):
            OUT_DTYPE = torch.float16
        else:
            print("Warning: fp16 is not supported on this GPU, use fp32 instead.")
    num_stages = 4 if OUT_DTYPE == torch.float16 else 3
    num_warps = 8
    if B >= (2048 * 16):
        num_warps = 16
    elif B >= (2048 * 8):
        num_warps = 8
    elif B >= (2048 * 4):
        num_warps = 4
    elif B >= 2048:
        num_warps = 2
    if "num_warps" in tuning_opt:
        num_warps = tuning_opt["num_warps"]
    if "num_stages" in tuning_opt:
        num_stages = tuning_opt["num_stages"]
    _fwd_kernel_aligned[(N + BLOCK_M - 1) // BLOCK_M, H, B](
        q, k, v, bias, sm_scale, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        bias.stride(0), bias.stride(1), bias.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BIAS_LAST_SIZE,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        OUT_DTYPE=OUT_DTYPE,
        IS_CAUSAL=is_
