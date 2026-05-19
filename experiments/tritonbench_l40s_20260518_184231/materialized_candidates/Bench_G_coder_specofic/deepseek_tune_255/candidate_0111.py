import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

@triton.jit
def _attn_fwd_inner(
    Q, K, V, sm_scale, m_i, l_i, qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
):
    # Matrix multiplication QK to get attention scores
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off_m in range(0, BLOCK_M, 16):
        for off_n in range(0, BLOCK_N, 16):
            qk += tl.dot(
                Q[off_m : off_m + BLOCK_M, :],
                tl.trans(K[off_n : off_n + BLOCK_N, :]),
            )
    qk *= sm_scale

    # Apply sliding window attention
    if SLIDING_WINDOW:
        m_ij = tl.maximum(m_i[:, None] - tl.arange(BLOCK_N), 0)
        qk -= m_ij
        l_ij = tl.math.exp2(l_i[:, None] - tl.arange(BLOCK_N))
        qk = tl.where(m_ij > 0, qk, float("-inf"))

    # Compute attention probabilities
    p = tl.math.exp2(qk)
    l_i_new = tl.math.log2(tl.sum(p, 1)) + tl.arange(BLOCK_N)
    alpha = tl.math.exp2(l_i - l_i_new[:, None])
    m_i_new = tl.maximum(tl.max(alpha, 0), m_i)
    l_i = tl.maximum(l_i_new, l_i)
    m_i = m_i_new

    # Compute attention output
    A = tl.dot(p.to(tl.float16), V)
    return Q, K, V, sm_scale, m_i, l_i, qk_scale, A, alpha

@triton.jit
def _attn_fwd(
    Q, K, V, sm_scale, m_i, l_i, Out,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr, BLOCK_M_PADDED: tl.constexpr,
    BLOCK_N_PADDED: tl.constexpr,
):
    qk_scale = tl.where(BLOCK_DMODEL > 128, 1.0, 1.0 / BLOCK_DMODEL)
    sm_scale *= qk_scale

    Q, K, V, sm_scale, m_i, l_i, _, Out, _ = _attn_fwd_inner(
        Q, K, V, sm_scale, m_i, l_i, qk_scale,
        BLOCK_M, BLOCK_N, BLOCK_DMODEL, SLIDING_WINDOW
    )

    Out += tl.dot(_, V)
    return Q, K, V, sm_scale, m_i, l_i, Out

class _forward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, k, v, sm_scale, m_i, l_i,
        out, atol, rtol, eps, sid, qk_scale,
        sliding_window, max_rank_for_layout_change,
        enable_mixed_bfloat16: bool = False,
        is_combined: bool = False,
    ):
        BLOCK_M_LIST = [16, 32, 64, 128]
        BLOCK_N_LIST = [16, 32, 64, 128]
        BLOCK_DMODEL_LIST = [64, 128, 256, 512]
        BLOCK_M_PADDED_LIST = [32, 64, 128, 256]
        BLOCK_N_PADDED_LIST = [32, 64, 128, 256]

        layout_change_count = torch.zeros([1], dtype=torch.int32, device="cuda")
        if sid["layout_change_count"] is not None:
            layout_change_count = sid["layout_change_count"]

        for BLOCK_M_PADDED in BLOCK_M_PADDED_LIST:
            for BLOCK_N_PADDED in BLOCK_N_PADDED_LIST:
                for BLOCK_M in BLOCK_M_LIST:
                    for BLOCK_N in BLOCK_N_LIST:
                        for BLOCK_DMODEL in BLOCK_DMODEL_LIST:
                            if (
                                BLOCK_M <= BLOCK_M_PADDED
                                and BLOCK_N <= BLOCK_N_PADDED
                                and BLOCK_DMODEL <= q.shape[1]
                                and q.stride(0) == 1
                                and k.stride(0) == 1
                                and v.stride(0) == 1
                                and out.stride(0) == 1
                            ):
                                q_arg = q.contiguous()
                                k_arg = k.contiguous()
                                v_arg = v.contiguous()
                                out_arg = out.contiguous()
                                if (
                                    q.dtype is torch.bfloat16
                                    and k.dtype is torch.bfloat16
                                    and v.dtype is torch.bfloat16
                                    and out.dtype is torch.bfloat16
                                ):
                                    pass
                                elif (
                                    q.dtype is torch.float16
                                    and k.dtype is torch.float16
                                    and v.dtype is torch.float16
                                    and out.dtype is torch.float16
                                ):
                                    pass
                                elif (
                                    q.dtype is torch.float16
                                    and k.dtype is torch.bfloat16
                                    and v.dtype is torch.float16
                                    and out.dtype is torch.float16
                                ):
                                    pass
                                elif (
                                    enable_mixed_bfloat16
                                    and q.dtype is torch.float16
                                    and k.dtype is torch.bfloat16
                                    and v.dtype is torch.float16
                                    and out.dtype is torch.float16
                                ):
                                    pass
                                else:
                                    continue

                                try:
                                    grid = lambda META: (
                                        triton.cdiv(q.shape[0], META["BLOCK_M"]),
                                    )
                                    _attn_fwd[grid](
                                        q_arg,
                                        k_arg,
                                        v_arg,
                                        sm_scale,
                                        m_i,
                                        l_i,
                                        out_arg,
                                        BLOCK_M=BLOCK_M,
                                        BLOCK_N=BLOCK_N,
                                        BLOCK_DMODEL=BLOCK_DMODEL,
                                        SLIDING_WINDOW=sliding_window,
                                        BLOCK_M_PADDED=BLOCK_M_PADDED,
                                        BLOCK_N_PADDED=BLOCK_N_PADDED,
                                        num_warps=8,
                                        num_stages=4,
                                    )
                                except triton.OutOfResources as e:
                                    if (
                                        layout_change_count >= max_rank_for_layout_change
                                    ):
                                        pass
                                    else:
                                        layout_change_count += 1
                                        sid["
