import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    Prob, V, Out,
    Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen,
    cur_batch, cur_head, cur_batch_seq_len, cur_batch_start_loc,
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    stride_ph, stride_pbs, stride_pb,
    stride_vbs, stride_vh, stride_vk, stride_vd,
    stride_obs, stride_oh, stride_ok, stride_od,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    cur_batch_end_loc = cur_batch_start_loc + cur_batch_seq_len
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    off_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_in_all_index = tl.load(B_Start_Loc + cur_batch + tl.arange(0, BLOCK_N))

    p_value = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    acc = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cur_batch_cur_index = cur_batch_start_loc + start_n + tl.arange(0, BLOCK_N)
        cur_batch_cur_mask = (cur_batch_cur_index < cur_batch_end_loc)[:, None]

        cur_block_req_idx = tl.load(B_req_idx + cur_batch_in_all_start_index + start_n + tl.arange(0, BLOCK_N))
        cur_block_token_offset = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_block_req_idx + stride_req_to_tokens_s * (cur_batch_cur_index - cur_batch_start_loc), mask=cur_batch_cur_mask, other=0)

        offs_n = start_n + tl.arange(0, BLOCK_N)
        offs_d = off_d
        block_v_offs = cur_batch * stride_vbs + cur_head * stride_vh + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vd
        v_value = tl.load(V + block_v_offs, mask=cur_batch_cur_mask, other=0.0)

        block_prob_offs = cur_batch * stride_pbs + cur_head * stride_ph + (offs_n) * stride_pb
        p_value = tl.load(Prob + block_prob_offs, mask=cur_batch_cur_mask[:, None], other=0.0)

        p_value = p_value * v_value
        acc += p_value

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = off_d
    acc = acc * p_value
    acc = tl.sum(acc, axis=0)

    off_k = tl.arange(0, BLOCK_DMODEL)
    block_out_offs = cur_batch * stride_obs + cur_head * stride_oh + off_k[None, :] * stride_od
    out_value = tl.load(Out + block_out_offs, mask=off_k[None, :] < BLOCK_DMODEL, other=0.0)
    acc = acc * out_value
    acc = tl.sum(acc, axis=1)

    acc = acc.to(Out.dtype.element_ty)
    off_k = tl.arange(0, BLOCK_DMODEL)
    block_out_offs = cur_batch * stride_obs + cur_head * stride_oh + off_k[None, :] * stride_od
    tl.store(Out + block_out_offs, acc, mask=off_k[None, :] < BLOCK_DMODEL)
    return

@torch.no_grad()
def token_att_fwd2(prob, v, out, req_to_token, b_req_idx, b_start_loc, b_seq_len):
    BLOCK = 128
    batch, head = b_seq_len.shape[0], v.shape[1]
    grid = (batch, head, triton.cdiv(prob.shape[2], BLOCK))

    kv_group_num = v.shape[1] // prob.shape[1]

    num_warps = 4
    dim = v.shape[-1]

    _fwd_kernel_token_att2[grid](
        prob, v, out,
        req_to_token, b_req_idx, b_start_loc, b_seq_len,
        0, 0, 0, 0,
        req_to_token.stride(0), req_to_token.stride(1),
        prob.stride(0), prob.stride(1), prob.stride(2),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_N=BLOCK,
        BLOCK_DMODEL=dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return
