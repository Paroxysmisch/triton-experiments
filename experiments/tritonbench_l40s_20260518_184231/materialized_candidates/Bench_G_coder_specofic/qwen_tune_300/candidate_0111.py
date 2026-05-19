import math
import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    acc,  # [BLOCK_M, BLOCK_DMODEL]
    l_i,  # [BLOCK_M]
    m_i,  # [BLOCK_M]
    q,    # [BLOCK_M, BLOCK_DMODEL]
    K_block_ptr,  # Pointer to K Block [BLOCK_K, BLOCK_DMODEL]
    V_block_ptr,  # Pointer to V Block [BLOCK_K, BLOCK_DMODEL]
    start_m,  # Starting chunk
    qk_scale,  # Scaling factor for qk
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
):
    # loop over k, v and update accumulator
    for start_n in range(0, BLOCK_N, BLOCK_M):
        start_n = tl.multiple_of(start_n, BLOCK_M)
        # -- compute qk ----
        k = tl.load(K_block_ptr + (start_n + tl.arange(0, BLOCK_M))[:, None] * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)[None, :])
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if STAGE == 1:
            qk += tl.dot(q, k)
        elif STAGE == 2:
            qk += tl.dot(q, k) * qk_scale
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(V_block_ptr + (start_n + tl.arange(0, BLOCK_M))[:, None] * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)[None, :])
        if SLIDING_WINDOW != -1:
            v_mask = (start_n + tl.arange(0, BLOCK_M)[:, None]) < SLIDING_WINDOW
            v = tl.where(v_mask, v, 0.0)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    return acc, l_i, m_i

def _attn_fwd(
    q,  # [N, BLOCK_M, BLOCK_DMODEL]
    k,  # [N, BLOCK_N, BLOCK_DMODEL]
    v,  # [N, BLOCK_N, BLOCK_DMODEL]
    o,  # [N, BLOCK_M, BLOCK_DMODEL]
    mask,  # [BLOCK_M, BLOCK_N]
    qk_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    STAGE: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
):
    # q [BLOCK_M, BLOCK_DMODEL]
    # k [BLOCK_N, BLOCK_DMODEL]
    # v [BLOCK_N, BLOCK_DMODEL]
    # accumulator [BLOCK_M, BLOCK_DMODEL]
    BLOCK_M = 128
    BLOCK_N = 64
    GROUP_M = 4
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=q.dtype)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    pid = tl.program_id(0)
    grid_m = tl.cdiv(q.shape[1], BLOCK_M)
    grid_n = tl.cdiv(k.shape[1], BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    # m [BLOCK_M, BLOCK_N]
    # q [BLOCK_M, BLOCK_DMODEL]
    # k [BLOCK_N, BLOCK_DMODEL]
    # acc [BLOCK_M, BLOCK_DMODEL]
    # qk [BLOCK_M, BLOCK_N]
    q_block_ptr = q + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    o_block_ptr = o + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    for start_n in range(0, grid_n, BLOCK_M):
        start_n = tl.multiple_of(start_n, BLOCK_M)
        K_block_ptr = k + start_n * BLOCK_DMODEL + tl.arange(0, BLOCK_N)[None, :] * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)[:, None]
        V_block_ptr = v + start_n * BLOCK_DMODEL + tl.arange(0, BLOCK_N)[None, :] * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)[:, None]
        if pid_n * BLOCK_M + BLOCK_M >= start_n and pid_n * BLOCK_M < start_n + BLOCK_N:
            # prefetch
            v = tl.load(V_block_ptr)
        else:
            v = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=v.dtype)
        q = tl.load(q_block_ptr)
        if STAGE & 1:
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_n, qk_scale, BLOCK_M, BLOCK_DMODEL, BLOCK_N, STAGE, SLIDING_WINDOW)
        if STAGE & 2:
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_n, qk_scale, BLOCK_M, BLOCK_DMODEL, BLOCK_N, STAGE, SLIDING_WINDOW)
        tl.store(o_block_ptr, acc.to(o.dtype.element_ty))
        q_block_ptr += BLOCK_M
        o_block_ptr += BLOCK_M

def _forward(
    q,
    k,
    v,
    o,
    mask,
    qk_scale,
    sparsity,
    end_step,
    BLOCK_M,
    BLOCK_N,
    BLOCK_DMODEL,
    STAGE,
    SLIDING_WINDOW,
):
    if sparsity > 0:
        raise RuntimeError("This model does not support SParsity")
    if mask is not None:
        raise RuntimeError("This model does not support Mask")
    if qk_scale is not None:
        raise RuntimeError("This model does not support Scaling")

    if end_step == -1:
        end_step = math.ceil(k.shape[1] / BLOCK_N)

    for stage in range(1, end_step + 1):
        _attn_fwd(q, k, v, o, mask, qk_scale, BLOCK_M, BLOCK_N, BLOCK_DMODEL, STAGE, SLIDING_WINDOW)
        if stage < end_step:
            q = tl.roll(q, -BLOCK_M, 1)
            k = tl.roll(k, -BLOCK_N, 1)
            v = tl.roll(v, -BLOCK_N, 1)
            o = tl.roll(o, -BLOCK_M, 1)
    return

@torch.no_grad()
def context_attention_fwd(q, k, v, o, mask, qk_scale, sparsity, end_step):
    assert q.shape[-1] == k.shape[-1]
    assert q.shape[-1] in {16, 32, 64, 128, 256, 512}
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert o.is_contiguous()
    # set default mask to None
    if mask is None:
        mask = torch.empty(0)
    # set default scale to None
    if qk_scale is None:
        qk_scale = math.sqrt(q.shape[-1])

    # run model
    BLOCK_M = 128
    BLOCK_N = 64
    GROUP_M = 4
    # stage 1: 1 <= end_step <= 3, BLOCK_M = 128, BLOCK_N = 64, GROUP_M = 4
    # stage 2: 4 <= end_step <= 6, BLOCK_M = 128, BLOCK_N = 64, GROUP_M = 8
    # stage 3: 7 <= end_step, BLOCK_M = 128, BLOCK_N = 64, GROUP_M = 8
    SLIDING_WINDOW = -1
    if end_step is None:
        end_step = 6

    if end_step <= 3:
        GROUP_M = 4
    if
