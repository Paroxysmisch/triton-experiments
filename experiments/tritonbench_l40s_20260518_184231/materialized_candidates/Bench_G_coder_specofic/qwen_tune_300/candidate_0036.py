import torch
import triton
import triton.language as tl
from packaging import version

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")

if TRITON_22:

    @triton.jit
    def _score_kernel(
        Q, K, M, sm_scale, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_oz, stride_oh, stride_om, stride_on, Z, H, N_CTX, scale,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, SLIDING_WINDOW: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        qvk_offset = off_hz * stride_qh
        Q_block_ptr = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
        K_block_ptr = tl.make_block_ptr(base=K + qvk_offset, shape=(BLOCK_DMODEL, N_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
        start_n = 0
        o_offset = off_hz * stride_oh
        Out_block_ptr = tl.make_block_ptr(base=Out + o_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        qk_scale = sm_scale * scale
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        q = tl.load(Q_block_ptr, boundary_check=(0, 1))
        q = (q * qk_scale).to(tl.float16)
        end_n = tl.cdiv(N_CTX, BLOCK_N)
        for start_n in range(0, end_n, 1):
            k = tl.load(K_block_ptr, boundary_check=(0, 1))
            qk += tl.dot(q, k)
            if SLIDING_WINDOW != -1:
                mask = offs_n + start_n * BLOCK_N < SLIDING_WINDOW
                qk = tl.where(mask, qk, float("-inf"))
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        qk = tl.where(offs_m[:, None] >= (offs_n[None, :] + start_n * BLOCK_N), qk, float("-inf"))
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        v = tl.load(K_block_ptr, boundary_check=(0, 1))
        p = p.to(tl.float16)
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new
        tl.store(Out_block_ptr, acc, boundary_check=(0, 1))

    @torch.inference_mode()
    def get_score(q, k, mask, sliding_window=-1):
        Lq, Lk = q.shape[-1], k.shape[-1]
        assert Lq == Lk
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        sm_scale = 1.0 / (Lk ** 0.5)
        scale = 1.0
        grid = lambda META: (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        BLOCK_M, BLOCK_N = 128, 64
        if BLOCK_M * BLOCK_N >= (16 * 1024):
            assert q.is_contiguous()
            _score_kernel[grid](
                q, k, mask, sm_scale, o, q.stride(0), q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3), o.stride(0), o.stride(1), o.stride(2),
                o.stride(3), q.shape[0], q.shape[1], q.shape[2], scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, SLIDING_WINDOW=sliding_window,
            )
        else:
            try:
                assert q.is_contiguous()
                _score_kernel[grid](
                    q, k, mask, sm_scale, o, q.stride(0), q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3), o.stride(0), o.stride(1), o.stride(2),
                    o.stride(3), q.shape[0], q.shape[1], q.shape[2], scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, SLIDING_WINDOW=sliding_window,
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    BLOCK_M, BLOCK_N = BLOCK_M // 2, BLOCK_N // 2
                    _score_kernel[grid](
                        q, k, mask, sm_scale, o, q.stride(0), q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3), o.stride(0), o.stride(1), o.stride(2),
                        o.stride(3), q.shape[0], q.shape[1], q.shape[2], scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, SLIDING_WINDOW=sliding_window,
                    )
                else:
                    raise e
        return o
else:
    pass
