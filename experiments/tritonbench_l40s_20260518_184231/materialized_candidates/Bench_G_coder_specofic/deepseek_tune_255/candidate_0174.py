import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att1(
    Q, K, B_Loc, B_Start_Loc, B_Seqlen,
    max_input_len,
    Att_Out,
    sm_scale,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_bh,
    stride_bl_bs, stride_bl_bh, stride_bl_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_M_PADDED: tl.constexpr, BLOCK_N_PADDED: tl.constexpr,
    num_warps: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_block_m = tl.program_id(2)

    start_k_loc = tl.load(B_Start_Loc + cur_batch * stride_bl_bh + cur_head * stride_bl_bh)
    seq_len = tl.load(B_Seqlen + cur_batch * stride_bl_bh)
    if seq_len <= 0:
        return

    offs_bl = cur_batch * stride_bl_bs + cur_head * stride_bl_bh + tl.arange(0, BLOCK_N)
    offs_bn = tl.arange(0, BLOCK_M_PADDED) + cur_block_m * BLOCK_M

    bl_m = tl.load(B_Loc + offs_bl, mask=offs_bn < seq_len, other=-1)
    bl_n = tl.load(B_Loc + start_k_loc + offs_bl, mask=offs_bn < seq_len, other=-1)

    offs_qm = cur_batch * stride_qbs + cur_head * stride_qh + offs_bn[None, :] * stride_qd
    offs_qk = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + offs_qm[:, None] + offs_qk[None, :]

    offs_km = bl_m[:, None] * stride_kbs + cur_head * stride_kh + offs_bn[None, :] * stride_kd
    offs_kn = bl_n[:, None] * stride_kbs + cur_head * stride_kh + offs_bn[None, :] * stride_kd
    k_ptrs = K + offs_km + offs_kn

    acc = tl.zeros((BLOCK_M_PADDED, BLOCK_N_PADDED), dtype=tl.float32)
    for d in range(0, BLOCK_DMODEL, 4):
        q = tl.load(q_ptrs + d)
        k = tl.load(k_ptrs + d)
        acc += tl.dot(q, k)

    acc = tl.where(offs_bn[:, None] < seq_len, acc, 0.0)
    acc = tl.where(tl.arange(0, BLOCK_M_PADDED)[None, :] < seq_len, acc, 0.0)

    acc = acc.to(tl.float16)
    acc *= sm_scale
    offs_am = cur_batch * stride_qbs + cur_head * stride_qh + offs_bn[None, :] * stride_qd
    offs_an = tl.arange(0, BLOCK_M_PADDED)[:, None]
    att_ptrs = Att_Out + offs_am + offs_an
    tl.store(att_ptrs, acc)


def token_att_fwd(q, k, att_out, b_loc, b_start_loc, b_seq_len, max_input_len):
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_DMODEL = q.shape[-1]
    assert BLOCK_DMODEL == k.shape[-1], f'{BLOCK_DMODEL} vs {k.shape[-1]}'
    assert BLOCK_DMODEL == att_out.shape[-1], f'{BLOCK_DMODEL} vs {att_out.shape[-1]}'
    BLOCK_M_PADDED = triton.next_power_of_2(BLOCK_M)
    BLOCK_N_PADDED = triton.next_power_of_2(BLOCK_N)

    num_warps = 4
    if BLOCK_DMODEL <= 128:
        num_warps = 8
    if BLOCK_DMODEL <= 64:
        num_warps = 16

    sm_scale = 1.0 / (BLOCK_DMODEL ** 0.5)
    if k.shape[-1] != k.stride(-1):
        k = k.contiguous()
    if q.shape[-1] != q.stride(-1):
        q = q.contiguous()
    _fwd_kernel_token_att1[
        q.shape[0] * q.shape[1] // BLOCK_N,
        q.shape[1],
    ] (
        q, k, b_loc, b_start_loc, b_seq_len, max_input_len,
        att_out, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        b_loc.stride(0), b_loc.stride(1),
        b_start_loc.stride(0), b_start_loc.stride(1), b_seq_len.stride(0), b_seq_len.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_DMODEL,
        BLOCK_M_PADDED, BLOCK_N_PADDED,
        num_warps=num_warps,
    )
    return
