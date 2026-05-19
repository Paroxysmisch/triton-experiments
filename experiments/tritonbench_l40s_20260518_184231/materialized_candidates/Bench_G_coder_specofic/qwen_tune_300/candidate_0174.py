import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att1(
    Q, K, B_Loc, B_Start_Loc, B_Seqlen, sm_scale, Att_Out,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_b_loc_b, stride_b_loc_s,
    stride_att_out_bs, stride_att_out_bh, stride_att_out_bl, stride_att_out_d,
    max_input_len,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)

    offs_d = tl.arange(0, 16)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    cur_batch_start_index = max_input_len - cur_batch_seq_len
    cur_batch_end_index = max_input_len

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)

    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + off_q + start_mark)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(B_Loc + stride_b_loc_b * cur_batch + stride_b_loc_s * offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
        off_k = k_loc[:, None] * stride_kbs + cur_head * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        off_o = cur_batch * stride_att_out_bs + cur_head * stride_att_out_bh + offs_n[None, :] * stride_att_out_bl + start_mark * stride_att_out_d
        off_k = k_loc[:, None]
        att_o_loc = off_k * stride_att_out_bl + off_o
        tl.store(Att_Out + att_o_loc, att_value, mask=offs_n_new[:, None] < cur_batch_end_index)
    return


@torch.no_grad()
def token_att_fwd(q, k, B_Loc, B_Start_Loc, B_Seqlen, max_input_len):
    BLOCK = 32
    assert q.shape[0] == B_Loc.shape[0] and q.shape[1] == k.shape[1], f"q shape {q.shape} k shape {k.shape}"
    sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    batch, head_num = B_Loc.shape[0], q.shape[1]
    att_out = torch.empty(batch, head_num, B_Loc.shape[-1], q.shape[-1], device="cuda")
    grid = (batch, head_num, triton.cdiv(max_input_len, BLOCK))

    _fwd_kernel_token_att1[grid](
        q, k, B_Loc, B_Start_Loc, B_Seqlen, sm_scale, att_out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        B_Loc.stride(0), B_Loc.stride(1),
        att_out.stride(0), att_out.stride(1), att_out.stride(2), att_out.stride(3),
        max_input_len,
        num_warps=4,
        BLOCK_N=BLOCK
    )
    return att_out
