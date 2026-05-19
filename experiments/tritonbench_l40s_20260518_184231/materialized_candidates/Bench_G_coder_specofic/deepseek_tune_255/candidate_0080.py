import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    Prob, V, Out,
    Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen,
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    stride_ph_b, stride_ph_h, stride_ph_d,
    stride_v_b, stride_v_h, stride_v_d,
    stride_out_b, stride_out_h, stride_out_s,
    stride_b_req_idx_b, stride_b_req_idx_s,
    stride_b_start_loc_b, stride_b_start_loc_s,
    stride_b_seqlen_b,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    kv_group_num: tl.constexpr,
    num_warps: tl.constexpr = 4,
    num_stages: tl.constexpr = 2,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch * stride_b_req_idx_b)
    cur_batch_start_index = tl.load(B_Start_Loc + cur_batch * stride_b_start_loc_b)

    off_p = cur_batch * stride_ph_b + cur_head * stride_ph_h + offs_d
    off_v = cur_batch * stride_v_b + cur_head * stride_v_h + offs_d
    off_req_tokens = cur_batch_req_idx * stride_req_to_tokens_b + offs_d
    off_out = cur_batch * stride_out_b + cur_head * stride_out_h + offs_d

    num_tokens = tl.cdiv(cur_batch_seq_len, BLOCK_N)
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for i in range(0, num_tokens, kv_group_num):
        offs_n = tl.arange(0, BLOCK_N)
        token_idx = cur_batch_start_index + offs_n * kv_group_num
        token_mask = token_idx < cur_batch_seq_len

        p = tl.load(
            Prob + off_p, mask=token_mask[:, None], other=0.0
        )  # [BLOCK_N, BLOCK_DMODEL]
        off_req_token = tl.load(
            Req_to_tokens + off_req_tokens, mask=token_mask[:, None], other=0.0
        )
        v = tl.load(
            V + off_v + off_req_token[:, :, None] * stride_v_d,
            mask=token_mask[:, None, None],
            other=0.0,
        )
        acc += tl.sum(p * v, 0)
        off_p += BLOCK_N * stride_ph_d
        off_v += BLOCK_N * stride_v_d
        off_req_tokens += BLOCK_N * stride_req_to_tokens_d

    acc = acc.to(tl.float16)
    tl.store(Out + off_out, acc)
    return

@torch.no_grad()
def token_att_fwd2(prob, v, out, req_to_tokens, b_req_idx, b_start_loc, b_seq_len, kv_group_num):
    seq_len = prob.shape[1]
    batch, head_num = b_seq_len.shape[0], v.shape[1]
    num_warps = 4
    assert seq_len % 32 == 0
    assert v.shape[1] == head_num
    assert v.shape[2] == req_to_tokens.shape[1]
    grid = (batch, head_num)
    num_stages = 2
    kwargs = [
        prob,
        v,
        out,
        req_to_tokens,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        req_to_tokens.stride(0),
        req_to_tokens.stride(1),
        prob.stride(0),
        prob.stride(1),
        prob.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        b_req_idx.stride(0),
        b_req_idx.stride(1),
        b_start_loc.stride(0),
        b_start_loc.stride(1),
        b_seq_len.stride(0),
    ]
    _fwd_kernel_token_att2[grid](
        num_warps=num_warps,
        num_stages=num_stages,
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=32,
        BLOCK_N=32,
        *kwargs,
    )
    return
