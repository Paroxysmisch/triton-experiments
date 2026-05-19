import torch
import triton
import triton.language as tl
from .triton_kernels import _attention_rel_h_rel_w_kernel_aligned_device

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, sm_scale,  # Q, K, V, B0: (batch, head, seq, head_dim)
    Rel_H, Rel_W,  # Rel_H, Rel_W: (1, head, rel_pos, head_dim)
    Out,  # Out: (batch, head, seq, head_dim)
    stride_qbs, stride_qh, stride_qd, stride_kbs, stride_kh, stride_kd, stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od, stride_rel_hbs, stride_rel_hh, stride_rel_hd, stride_rel_wbs, stride_rel_wh,
    stride_rel_whd, rel_h_start_idx, rel_w_start_idx,
    Z, H, N_CTX, P_SEQ,  # Z: batch, H: head, N_CTX: seq_len, P_SEQ: prefix_seq_len
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_DMODEL_PADDED: tl.constexpr,
    IS_CAUSAL: tl.constexpr, CHECK_SEQ: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_batch_seq_len = tl.where(cur_batch < Z - P_SEQ, N_CTX, N_CTX - P_SEQ)
    cur_batch_prefix_len = tl.where(cur_batch < Z - P_SEQ, 0, P_SEQ)

    block_start_loc = BLOCK_M * start_m

    # compute relative positional encoding index
    rel_pos = block_start_loc - cur_batch_seq_len  # rel_pos: 0, -1, -2, ..., -cur_batch_seq_len

    # compute pad
    if IS_CAUSAL:
        if BLOCK_M == BLOCK_N:
            pad_m = 0
        else:
            pad_m = BLOCK_N - BLOCK_M
    else:
        pad_m = 0

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (cur_batch * stride_qbs + cur_head * stride_qh) \
            + (offs_m[:, None] * stride_qd + offs_d[None, :])
    off_k = (cur_batch * stride_kbs + cur_head * stride_kh) \
            + (offs_n[:, None] * stride_kd + offs_d[None, :])
    off_v = (cur_batch * stride_vbs + cur_head * stride_vh) \
            + (offs_n[:, None] * stride_vd + offs_d[None, :])
    off_b0 = (cur_batch * stride_qbs + cur_head * stride_qh) \
             + (offs_m[:, None] * stride_qd + offs_n[None, :])
    off_rel_h = (cur_head * stride_rel_hh
                 + (rel_h_start_idx + rel_pos) * stride_rel_hd
                 + offs_d[None, :])
    off_rel_w = (cur_head * stride_rel_wh
                 + (rel_w_start_idx + rel_pos) * stride_rel_whd
                 + offs_d[None, :])

    q = tl.load(Q + off_q)
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    b0 = tl.load(B0 + off_b0, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)

    # causal: mask out future tokens
    if IS_CAUSAL:
        causal_mask = (cur_batch_seq_len - offs_m[:, None]) > offs_n[None, :]
        k_ptrs = tl.where(causal_mask, k_ptrs, 0)
        v_ptrs = tl.where(causal_mask, v_ptrs, 0)
        b0 = tl.where(causal_mask, b0, 0.0)

    # compute attention
    for start_n in range(0, block_start_loc + cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(k_ptrs + start_n * stride_kd)
        # mask = tl.load(mask_ptrs + start_n, mask=(start_n + offs_n) < cur_batch_seq_len, other=0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        # -- compute m_ij, p, l_ij
        if CHECK_SEQ:
            qk = tl.where((cur_batch_seq_len - offs_m[:, None]) >= (start_n + offs_n[None, :]), qk, -100000000.0)
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
        v = tl.load(v_ptrs + start_n * stride_vd, mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0)
        # v = tl.load(v_ptrs + start_n * stride_vd)
        p = p.to(v.dtype)
        rel_h = tl.load(Rel_H + off_rel_h, mask=(rel_pos + offs_m[:, None]) >= 0, other=0)
        rel_w = tl.load(Rel_W + off_rel_w, mask=(rel_pos + offs_n[None, :]) >= 0, other=0)
        acc += tl.dot(p, v) + b0 * (tl.dot(p, rel_h) + tl.dot(p_scale[:, None], rel_w))
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    off_o = (cur_batch * stride_obs + cur_head * stride_oh) \
            + (offs_m[:, None] * stride_od + offs_d[None, :])
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc.to(Out.type.element_ty))

    if pad_m > 0:
        off_pad = (cur_batch * stride_obs + cur_head * stride_oh) \
                  + ((offs_m + pad_m)[:, None] * stride_od + offs_d[None, :])
        out_pad_ptrs = Out + off_pad
        tl.store(out_pad_ptrs, 0.0)

def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, b0, rel_h, rel_w, causal, check_seq, output,
                                                 rel_h_start_idx, rel_w_start_idx, max_input_len, prefix_seq_len,
                                                 block_dmodel, is_bfloat16):
    batch, head, seq_len = q.shape
    assert k.shape == q.shape and v.shape == q.shape and b0.shape == q.shape
    assert rel_h.shape == (1, head, -1, block_dmodel) and rel_w.shape == rel_h.shape
    assert output.shape == q.shape
    if is_bfloat16:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        b0 = b0.to(torch.bfloat16)
        rel_h = rel_h.to(torch.bfloat16)
        rel_w = rel_w.to(torch.bfloat16)
        output = output.to(torch.bfloat16)
    BLOCK_M = BLOCK_N = block_dmodel
    if block_dmodel <= 64:
        BLOCK_M = BLOCK_N = 64
    elif block_dmodel <= 128:
        BLOCK_M = BLOCK_N = 128
    elif block_dmodel <= 256:
        BLOCK_M = BLOCK_N = 256
    GROUP_M = 1
    if BLOCK_M >= 128:
        GROUP_M = 8
    if BLOCK_M >= 512:
        GROUP_M = 16
    if BLOCK_M >= 1024:
        GROUP_M = 32
    num_warps = 4 if block_dmodel <= 64 else 8
    if block_dmodel > 128:
        num_warps = 16
