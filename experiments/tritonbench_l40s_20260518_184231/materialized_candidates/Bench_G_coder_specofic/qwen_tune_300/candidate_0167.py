import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_ptrs,
    K_scale_ptr,
    V_ptrs,
    start_n,
    qk_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
    N_CTX: tl.constexpr,
):
    # q has shape (BLOCK_M, D_HEAD)
    q = q.to(tl.float32)
    # The compiler can't reason about what's inside these pointers, so we load them explicitly.
    # We also can't reason about the shape of K, so we load BLOCK_N elements at a time.
    K = tl.load(K_ptrs, mask=K_scale_ptr is not None, other=0.0)
    V = tl.load(V_ptrs)
    if K_scale_ptr is not None:
        K_scale = tl.load(K_scale_ptr)
        K = (K * K_scale).to(K.dtype)
    # `qk` has shape (BLOCK_M, BLOCK_N)
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, K)
    # Work on a cache line at a time.
    qk += tl.summation(qk, axis=1)[:, None]
    qk *= qk_scale
    # `m_i` has shape (BLOCK_M,)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    # This is equivalent to `np.exp(m_i - m_i_new)`, but more numerically stable.
    l_i_new = l_i * tl.exp(m_i - m_i_new)
    # `p` has shape (BLOCK_M, BLOCK_N)
    qk = qk - m_i_new[:, None]
    qk = tl.softmax(qk)
    # `l_i_new` has shape (BLOCK_M,)
    l_i_new += tl.sum(qk, 1)
    # The attention output is `qk @ V`, but we don't store `qk` explicitly.
    # We just accumulate the results of `qk @ V` into `acc`.
    # `acc` has shape (BLOCK_M, D_HEAD)
    acc = acc * l_i + tl.dot(qk.to(V.dtype), V)
    # `l_i` and `m_i` become `l_i_new` and `m_i_new` for the next iteration.
    l_i = l_i_new
    m_i = m_i_new
    return acc, l_i, m_i

@torch.inference_mode()
def _attn_fwd(
    q,
    k,
    v,
    q_scale,
    k_scale,
    out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    qk_scale,
    N_CTX,
):
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = out
    # reshape input data into 2D tensors
    q = q.reshape((q.shape[0] * q.shape[1], q.shape[2]))
    k = k.reshape((k.shape[0] * k.shape[1], k.shape[2]))
    v = v.reshape((v.shape[0] * v.shape[1], v.shape[2]))
    o = o.reshape((o.shape[0] * o.shape[1], o.shape[2]))
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // q.shape[-1]
    BLOCK_M = min(MAX_FUSED_SIZE, triton.next_power_of_2(q.shape[0]))
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(k.shape[0]))
    # heuristics for number of warps
    num_warps = min(max(BLOCK_M // 256, 1), 8)
    # enqueue kernel
    grid = (triton.cdiv(q.shape[0], BLOCK_M), triton.cdiv(k.shape[0], BLOCK_N))
    _attn_fwd_inner[grid](
        o,
        q,
        k,
        v,
        q_scale,
        k_scale,
        qk_scale,
        N_CTX,
        BLOCK_M,
        BLOCK_N,
        q.shape[-1],
        num_warps=num_warps,
    )
    # restore output to original shape
    out = o.reshape((stride_qz, stride_qh, q.shape[0], q.shape[1]))
    return out
