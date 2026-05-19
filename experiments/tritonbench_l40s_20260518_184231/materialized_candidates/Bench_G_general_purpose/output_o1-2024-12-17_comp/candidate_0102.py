import torch
import triton
import triton.language as tl

@triton.jit
def parallel_retention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    nheads, seqlen, dim_head,
    stride_qh, stride_qs, stride_qd,
    stride_kh, stride_ks, stride_kd,
    stride_vh, stride_vs, stride_vd,
    stride_oh, stride_os, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    head_idx = pid_h

    decay_factor = 1.0 / (1.0 + head_idx * 0.01)
    scale = 1.0 / (dim_head**0.5)

    q_ptrs = Q_ptr + head_idx * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    mask_q = (offs_m < seqlen)
    mask_d = (offs_d < dim_head)
    q += tl.where(mask_q[:, None] & mask_d[None, :], tl.load(q_ptrs, mask=mask_q[:, None] & mask_d[None, :], other=0.0), 0.0)

    o_acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    for s_i in range(0, seqlen, BLOCK_M):
        k_offs_m = s_i + tl.arange(0, BLOCK_M)
        k_ptrs = K_ptr + head_idx * stride_kh + k_offs_m[:, None] * stride_ks + offs_d[None, :] * stride_kd
        src_mask = (k_offs_m < seqlen)
        k = tl.where(src_mask[:, None] & mask_d[None, :], tl.load(k_ptrs, mask=src_mask[:, None] & mask_d[None, :], other=0.0), 0.0)
        att = tl.dot(q, tl.trans(k)) * scale * decay_factor
        v_ptrs = V_ptr + head_idx * stride_vh + k_offs_m[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.where(src_mask[:, None] & mask_d[None, :], tl.load(v_ptrs, mask=src_mask[:, None] & mask_d[None, :], other=0.0), 0.0)
        o_acc += tl.dot(att, v)

    o_ptrs = O_ptr + head_idx * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o_acc, mask=mask_q[:, None] & mask_d[None, :])

@triton.jit
def _parallel_retention_bwd_dq(
    Q_ptr, K_ptr, V_ptr, DO_ptr, DQ_ptr,
    nheads, seqlen, dim_head,
    stride_qh, stride_qs, stride_qd,
    stride_kh, stride_ks, stride_kd,
    stride_vh, stride_vs, stride_vd,
    stride_doh, stride_dos, stride_dod,
    stride_dqh, stride_dqs, stride_dqd,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    head_idx = pid_h

    decay_factor = 1.0 / (1.0 + head_idx * 0.01)
    scale = 1.0 / (dim_head**0.5)

    do_ptrs = DO_ptr + head_idx * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    do_val = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    mask_m = (offs_m < seqlen)
    mask_d = (offs_d < dim_head)
    do_val += tl.where(mask_m[:, None] & mask_d[None, :], tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0), 0.0)

    dq_acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    for s_i in range(0, seqlen, BLOCK_M):
        k_offs_m = s_i + tl.arange(0, BLOCK_M)
        src_mask = (k_offs_m < seqlen)
        k_ptrs = K_ptr + head_idx * stride_kh + k_offs_m[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_val = tl.where(src_mask[:, None] & mask_d[None, :], tl.load(k_ptrs, mask=src_mask[:, None] & mask_d[None, :], other=0.0), 0.0)
        v_ptrs = V_ptr + head_idx * stride_vh + k_offs_m[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v_val = tl.where(src_mask[:, None] & mask_d[None, :], tl.load(v_ptrs, mask=src_mask[:, None] & mask_d[None, :], other=0.0), 0.0)
        att_grad = tl.dot(do_val, tl.trans(v_val)) * scale * decay_factor
        dq_acc += tl.dot(att_grad, k_val)

    dq_ptrs = DQ_ptr + head_idx * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq_acc, mask=mask_m[:, None] & mask_d[None, :])

@triton.jit
def _parallel_retention_bwd_dkv(
    Q_ptr, K_ptr, V_ptr, DO_ptr, DK_ptr, DV_ptr,
    nheads, seqlen, dim_head,
    stride_qh, stride_qs, stride_qd,
    stride_kh, stride_ks, stride_kd,
    stride_vh, stride_vs, stride_vd,
    stride_doh, stride_dos, stride_dod,
    stride_dkh, stride_dks, stride_dkd,
    stride_dvh, stride_dvs, stride_dvd,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    head_idx = pid_h

    decay_factor = 1.0 / (1.0 + head_idx * 0.01)
    scale = 1.0 / (dim_head**0.5)

    k_ptrs = K_ptr + head_idx * stride_kh + offs_m[:, None] * stride_ks + offs_d[None, :] * stride_kd
    k_val = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    mask_m = (offs_m < seqlen)
    mask_d = (offs_d < dim_head)
    k_val += tl.where(mask_m[:, None] & mask_d[None, :], tl.load(k_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0), 0.0)

    v_ptrs = V_ptr + head_idx * stride_vh + offs_m[:, None] * stride_vs + offs_d[None, :] * stride_vd
    v_val = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    v_val += tl.where(mask_m[:, None] & mask_d[None, :], tl.load(v_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0), 0.0)

    dk_acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    for s_i in range(0, seqlen, BLOCK_M):
        q_offs_m = s_i + tl.arange(0, BLOCK_M)
        src_mask = (q_offs_m < seqlen)
        q_ptrs = Q_ptr + head_idx * stride_qh + q_offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        q_val = tl.where(src_mask[:, None] & mask_d[None, :], tl.load(q_ptrs, mask=src_mask[:, None] & mask_d[None, :], other=0.0), 0.0)
        do_ptrs = DO_ptr + head_idx * stride_doh + q_offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        do_val = tl.where(src_mask[:, None] & mask_d[None, :], tl.load(do_ptrs, mask=src_mask[:, None] & mask_d[None, :], other=0.0), 0.0)
        att = tl.dot(q_val, tl.trans(k_val)) * scale * decay_factor
        dk_acc += tl.dot(tl.trans(q_val), tl.dot(do_val, tl.trans(v_val)) * scale * decay_factor)
        dv_acc += tl.dot(tl.trans(att), do_val)

    dk_ptrs = DK_ptr + head_idx * stride_dkh + offs_m[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = DV_ptr + head_idx * stride_dvh + offs_m[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    tl.store(dk_ptrs, dk_acc, mask=mask_m[:, None] & mask_d[None, :])
    tl.store(dv_ptrs, dv_acc, mask=mask_m[:, None] & mask_d[None, :])

class ParallelRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        B, H, S, D = q.shape
        o = torch.empty_like(q)

        grid =
