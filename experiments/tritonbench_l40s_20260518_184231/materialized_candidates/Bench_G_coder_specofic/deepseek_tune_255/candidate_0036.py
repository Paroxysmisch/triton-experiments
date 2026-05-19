import torch
import triton
import triton.language as tl

@triton.jit
def _score_kernel(
    Q,
    K,
    M,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_oz,
    stride_oh,
    stride_om,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_M_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    o_offset = off_hz * stride_oh

    z_block_ct = tl.cdiv(Z, BLOCK_M_N)
    z_idx = start_m // z_block_ct
    z_range_start = z_idx * BLOCK_M_N
    z_range_end = tl.min(z_range_start + BLOCK_M_N, Z)
    n_ctx_range_start = tl.zeros((1,), tl.int32)
    n_ctx_range_end = tl.zeros((1,), tl.int32) + N_CTX

    if SLIDING_WINDOW:
        n_ctx_range_start = tl.maximum(n_ctx_range_start, start_m * BLOCK_M)
        n_ctx_range_end = tl.minimum(n_ctx_range_end, (start_m + 1) * BLOCK_M)

    while z_idx < z_block_ct:
        q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(Z, H, N_CTX),
            strides=(stride_qz, stride_qh, stride_qm),
            offsets=(z_range_start, 0, n_ctx_range_start),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 2),
        )
        k_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(H, N_CTX, Z),
            strides=(stride_kh, stride_kn, stride_kz),
            offsets=(0, n_ctx_range_start, z_range_start),
            block_shape=(BLOCK_N, BLOCK_M),
            order=(0, 1),
        )
        q = tl.load(q_block_ptr)
        q = (q.to(tl.float32) - 128) / 128
        k = tl.load(k_block_ptr)
        k = (k.to(tl.float32) - 128) / 128
        qk = tl.dot(q, k)

        if HAS_MASK:
            m_val = tl.load(M + q_offset + n_ctx_range_start + z_range_start)
            m_mask = m_val.to(tl.float32)
            qk = tl.where(m_mask.to(tl.bool), qk, float("-inf"))

        qk = tl.where(qk == float("-inf"), qk, qk * 100)
        qk = tl.where(qk == float("-inf"), qk, qk - tl.max(qk, axis=1, keepdims=True))

        sm_scale = 100 / tl.sum(tl.exp(qk), axis=1, keepdims=True)
        o = tl.exp(qk) * sm_scale
        o = o.to(tl.float16)

        o_block_ptr = tl.make_block_ptr(
            base=Out + o_offset,
            shape=(Z, H, N_CTX),
            strides=(stride_oz, stride_oh, stride_om),
            offsets=(z_range_start, 0, n_ctx_range_start),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 2),
        )
        tl.store(o_block_ptr, o)

        z_idx += 1
        start_m += z_block_ct
        z_idx = start_m // z_block_ct
        z_range_start = z_idx * BLOCK_M_N
        z_range_end = tl.min(z_range_start + BLOCK_M_N, Z)
        n_ctx_range_start = tl.maximum(n_ctx_range_start - BLOCK_M, 0)
        n_ctx_range_end = tl.minimum(n_ctx_range_end + BLOCK_M, N_CTX)


def get_score(
    q,
    k,
    m=None,
    out=None,
    slid_win=False,
    BLOCK_M=16,
    BLOCK_N=16,
):
    if m is None:
        MASK = False
    else:
        MASK = True

    if out is None:
        out = torch.empty_like(q)

    assert q.dim() == k.dim() == 3
    assert q.size(0) == k.size(0)
    assert q.size(2) == k.size(2)
    assert q.size(1) == k.size(1)

    if BLOCK_N > q.size(1):
        BLOCK_N = q.size(1)

    if BLOCK_M > q.size(2):
        BLOCK_M = q.size(2)

    if BLOCK_N % 16 != 0 or BLOCK_M % 16 != 0:
        BLOCK_N = (BLOCK_N // 16) * 16
        BLOCK_M = (BLOCK_M // 16) * 16

    if BLOCK_N > q.size(1) or BLOCK_M > q.size(2):
        return get_score(
            q,
            k,
            m,
            out,
            slid_win,
            BLOCK_M=BLOCK_M // 2,
            BLOCK_N=BLOCK_N // 2,
        )

    try:
        _score_kernel[(q.size(0) * q.size(1) * q.size(2))](
            q,
            k,
            m,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            q.size(0),
            q.size(1),
            q.size(2),
            BLOCK_M,
            BLOCK_N,
            slid_win,
            MASK,
            BLOCK_M_N=BLOCK_M,
            num_warps=8,
            num_stages=4,
        )
    except RuntimeError as e:
        if "resource constraint" in str(e):
            return get_score(
                q,
                k,
                m,
                out,
                slid_win,
                BLOCK_M=BLOCK_M // 2,
                BLOCK_N=BLOCK_N // 2,
            )
        else:
            raise e
    return out
