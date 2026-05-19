import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    B, H, E, D,
    stride_qb, stride_qh, stride_qe, stride_qd,
    stride_kb, stride_kh, stride_ke, stride_kd,
    stride_vb, stride_vh, stride_ve, stride_vd,
    stride_ob, stride_oh, stride_oe, stride_od,
    BLOCK: tl.constexpr, NUM_BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    b_idx = pid // H
    h_idx = pid % H
    e_block_idx = bid

    e_start = e_block_idx * BLOCK
    offs_d = tl.arange(0, BLOCK)
    offs_dd = tl.arange(0, BLOCK)
    q_offset = b_idx * stride_qb + h_idx * stride_qh
    o_offset = b_idx * stride_ob + h_idx * stride_oh

    Q_block = tl.load(
        Q_ptr + q_offset + (e_start + offs_d)[:, None] * stride_qe + offs_dd[None, :] * stride_qd,
        mask=(e_start + offs_d[:, None] < E) & (offs_dd[None, :] < D),
        other=0.0
    )

    acc = tl.zeros([BLOCK, D], dtype=tl.float32)
    k_offset = b_idx * stride_kb + h_idx * stride_kh

    for nb in range(NUM_BLOCK):
        e_offset = nb * BLOCK
        K_block = tl.load(
            K_ptr + k_offset + (e_offset + offs_d)[:, None] * stride_ke + offs_dd[None, :] * stride_kd,
            mask=(e_offset + offs_d[:, None] < E) & (offs_dd[None, :] < D),
            other=0.0
        )
        score = tl.dot(Q_block, tl.trans(K_block))
        v_offset = b_idx * stride_vb + h_idx * stride_vh
        V_block = tl.load(
            V_ptr + v_offset + (e_offset + offs_d)[:, None] * stride_ve + offs_dd[None, :] * stride_vd,
            mask=(e_offset + offs_d[:, None] < E) & (offs_dd[None, :] < D),
            other=0.0
        )
        attn_out = tl.dot(score.to(Q_block.dtype), V_block)
        acc += attn_out

    tl.store(
        O_ptr + o_offset + (e_start + offs_d)[:, None] * stride_oe + offs_dd[None, :] * stride_od,
        acc,
        mask=(e_start + offs_d[:, None] < E) & (offs_dd[None, :] < D)
    )


@triton.jit
def _bwd_intra_kernel(
    DO_ptr,
    Q_ptr, K_ptr, V_ptr,
    DQ_ptr, DK_ptr, DV_ptr,
    B, H, E, D,
    stride_dob, stride_doh, stride_doe, stride_dod,
    stride_qb, stride_qh, stride_qe, stride_qd,
    stride_kb, stride_kh, stride_ke, stride_kd,
    stride_vb, stride_vh, stride_ve, stride_vd,
    stride_dqb, stride_dqh, stride_dqe, stride_dqd,
    stride_dkb, stride_dkh, stride_dke, stride_dkd,
    stride_dvb, stride_dvh, stride_dve, stride_dvd,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    b_idx = pid // H
    h_idx = pid % H
    e_block_idx = bid

    e_start = e_block_idx * BLOCK
    offs_d = tl.arange(0, BLOCK)
    offs_dd = tl.arange(0, BLOCK)
    do_offset = b_idx * stride_dob + h_idx * stride_doh

    DO_block = tl.load(
        DO_ptr + do_offset + (e_start + offs_d)[:, None] * stride_doe + offs_dd[None, :] * stride_dod,
        mask=(e_start + offs_d[:, None] < E) & (offs_dd[None, :] < D),
        other=0.0
    )

    q_offset = b_idx * stride_qb + h_idx * stride_qh
    Q_block = tl.load(
        Q_ptr + q_offset + (e_start + offs_d)[:, None] * stride_qe + offs_dd[None, :] * stride_qd,
        mask=(e_start + offs_d[:, None] < E) & (offs_dd[None, :] < D),
        other=0.0
    )

    DQ_block = DO_block
    tl.store(
        DQ_ptr + b_idx * stride_dqb + h_idx * stride_dqh + (e_start + offs_d)[:, None] * stride_dqe + offs_dd[None, :] * stride_dqd,
        DQ_block,
        mask=(e_start + offs_d[:, None] < E) & (offs_dd[None, :] < D)
    )


@triton.jit
def _bwd_inter_kernel(
    Q_ptr, K_ptr, V_ptr,
    DQ_ptr, DK_ptr, DV_ptr,
    B, H, E, D,
    stride_qb, stride_qh, stride_qe, stride_qd,
    stride_kb, stride_kh, stride_ke, stride_kd,
    stride_vb, stride_vh, stride_ve, stride_vd,
    stride_dqb, stride_dqh, stride_dqe, stride_dqd,
    stride_dkb, stride_dkh, stride_dke, stride_dkd,
    stride_dvb, stride_dvh, stride_dve, stride_dvd,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    b_idx = pid // H
    h_idx = pid % H
    pass


class LightningAttention2NoDecayFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        B, H, E, D = Q.shape
        BLOCK = 64
        NUM_BLOCK = (E + BLOCK - 1) // BLOCK
        O = torch.empty_like(Q)
        grid = (B * H, NUM_BLOCK)
        _fwd_kernel[grid](
            Q, K, V, O,
            B, H, E, D,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK
        )
        ctx.save_for_backward(Q, K, V)
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V = ctx.saved_tensors
        B, H, E, D = Q.shape
        DQ = torch.zeros_like(Q)
        DK = torch.zeros_like(K)
        DV = torch.zeros_like(V)
        BLOCK = 64
        CBLOCK = 32
        NUM_BLOCK = (E + BLOCK - 1) // BLOCK
        NUM_CBLOCK = (E + CBLOCK - 1) // CBLOCK

        grid_intra = (B * H, NUM_BLOCK)
        _bwd_intra_kernel[grid_intra](
            dO,
            Q, K, V,
            DQ, DK, DV,
            B, H, E, D,
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            DQ.stride(0), DQ.stride(1), DQ.stride(2), DQ.stride(3),
            DK.stride(0), DK.stride(1), DK.stride(2), DK.stride(3),
            DV.stride(0), DV.stride(1), DV.stride(2), DV.stride(3),
            BLOCK=BLOCK
        )

        grid_inter = (B * H,)
        _bwd_inter_kernel[grid_inter](
            Q, K, V,
            DQ, DK, DV,
            B, H, E, D,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            DQ.stride(0), DQ.stride(1), DQ.stride(2), DQ.stride(3),
            DK.stride(0), DK.stride(1), DK.stride(2), DK.stride(3),
            DV.stride(0), DV.stride(1), DV.stride(2), DV.stride(3),
            BLOCK=BLOCK
        )
        return DQ, DK, DV


def lightning_attention_2_no_decay(Q, K, V):
    return LightningAttention2NoDecayFunction.apply(Q, K, V)
