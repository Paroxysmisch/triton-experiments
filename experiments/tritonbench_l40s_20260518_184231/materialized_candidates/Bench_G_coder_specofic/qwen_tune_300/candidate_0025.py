import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,  # Input buffers
    L, O,  # Output buffers
    L_max,  # Informational buffer
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_lz, stride_lh, stride_lm,  # Stride for L_max
    Z, H,  # Metadata
    BLOCK_M: tl.constexpr,  # Block sizes
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(Z, H, BLOCK_M, BLOCK_N),
        strides=(stride_qz, stride_qh, stride_qm, stride_qk),
        offsets=(0, 0, start_m * BLOCK_M, 0),
        block_shape=(1, 1, BLOCK_M, BLOCK_N),
        order=(3, 2, 1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(Z, H, BLOCK_N, BLOCK_M),
        strides=(stride_kz, stride_kh, stride_kn, stride_kk),
        offsets=(0, 0, 0, 0),
        block_shape=(1, 1, BLOCK_N, BLOCK_M),
        order=(3, 2, 1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(Z, H, BLOCK_M, BLOCK_N),
        strides=(stride_vz, stride_vh, stride_vk, stride_vn),
        offsets=(0, 0, 0, 0),
        block_shape=(1, 1, BLOCK_M, BLOCK_N),
        order=(3, 2, 1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_offset,
        shape=(Z, H, BLOCK_M, BLOCK_N),
        strides=(stride_oz, stride_oh, stride_om, stride_on),
        offsets=(0, 0, start_m * BLOCK_M, 0),
        block_shape=(1, 1, BLOCK_M, BLOCK_N),
        order=(3, 2, 1, 0),
    )
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    qk_exps = tl.arange(0, BLOCK_M) * qk_scale
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.float16)
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else (Z * H * BLOCK_M)
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, k)
        qk += qk_exps[None, :]
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        K_block_ptr = tl.advance(K_block_ptr, (0, 0, BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (0, 0, 0, BLOCK_N))
    acc = acc / l_i[:, None]
    o = acc.to(tl.float16)
    l_max = m_i + tl.math.log2(l_i)
    L_block_ptr = tl.make_block_ptr(
        base=L + off_hz * stride_lh,
        shape=(Z, H, BLOCK_M),
        strides=(stride_lz, stride_lh, stride_lm),
        offsets=(0, 0, start_m * BLOCK_M),
        block_shape=(1, 1, BLOCK_M),
        order=(2, 1, 0),
    )
    tl.store(L_block_ptr, l_max[:, None, None])
    tl.store(O_block_ptr, o)

def flash_attn_triton(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sm_scale: float,
    causal: Optional[bool] = False,
) -> Tensor:
    batch, heads, seq_len, d_head = q.shape
    assert k.shape == (batch, heads, seq_len, d_head)
    assert v.shape == (batch, heads, seq_len, d_head)
    assert d_head in {16, 32, 64, 128}
    # We always want to promote to float32 here since we will be doing math
    if q.dtype == torch.float16:
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
    # This is a codegen hack to get the right speed for causal vs non-causal
    # It should compile to basically the same code, just with different masks set
    # to zero.
    causal = causal and True  # type: ignore
    BLOCK_M = 128
    BLOCK_N = 64 if d_head <= 64 else 32
    num_warps = 4 if d_head <= 64 else 8
    o = torch.empty_like(q)
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads, 1)
    L_max = torch.empty((batch, heads, BLOCK_M), device=q.device, dtype=torch.float32)
    _fwd_kernel[grid](
        q, k, v, sm_scale,  # type: ignore
        o, L_max,  # type: ignore
        stride_qm=q.stride(2), stride_qk=q.stride(3),
        stride_kn=k.stride(2), stride_kk=k.stride(3),
        stride_vk=v.stride(2), stride_vn=v.stride(3),
        stride_om=o.stride(2), stride_on=o.stride(3),
        stride_lm=L_max.stride(1),
        Z=batch,
        H=heads,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=causal,
        num_warps=num_warps,
        num_stages=1,
    )
    o /= sm_scale
    return o
