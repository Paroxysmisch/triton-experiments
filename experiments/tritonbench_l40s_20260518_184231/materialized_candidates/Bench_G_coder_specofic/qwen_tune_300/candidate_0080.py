import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    Prob, V, Out, Req_to_tokens,
    B_req_idx, B_Start_Loc, B_Seqlen,
    cur_batch, cur_head, 
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    stride_ph, stride_pbs, stride_pb_seq,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = 0
    
    v_loc_off = cur_batch_req_idx * stride_req_to_tokens_b + (cur_batch_start_index + offs_n) * stride_req_to_tokens_s
    p_offs = cur_head * stride_ph + (cur_batch * stride_pbs + cur_batch_start_index * stride_pb_seq + offs_n * stride_pb_seq) 
    v_offs = cur_head * stride_vh + offs_d[None, :] * stride_vd

    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    V_loc = tl.load(Req_to_tokens + v_loc_off, mask=(cur_batch_start_index + offs_n) < cur_batch_seq_len, other=0)
    
    for start_mark in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = start_mark
        p_value = tl.load(Prob + p_offs + start_n * stride_pb_seq, mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0)
        v_loc = tl.load(Req_to_tokens + v_loc_off + start_n * stride_req_to_tokens_s, mask=(start_n + offs_n) < cur_batch_seq_len, other=0)
        v_value = tl.load(V + v_offs + v_loc[:, None] * stride_vbs, mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)
        acc += tl.sum(p_value[:, None] * v_value, 0)

    acc = acc.to(tl.float16)
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)
    return

@torch.no_grad()
def token_att_fwd2(prob, v, out, req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen):
    seq_len = B_Seqlen.shape[1]
    kv_group_num = triton.cdiv(seq_len, v.shape[2])
    
    grid = (B_req_idx.shape[0], v.shape[1], 1)
    num_warps = 1
    num_stages = 2

    _fwd_kernel_token_att2[grid](
        prob, v, out, req_to_tokens,
        B_req_idx, B_Start_Loc, B_Seqlen,
        0, 0,
        req_to_tokens.stride(0), req_to_tokens.stride(1),
        prob.stride(0), prob.stride(1), prob.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_DMODEL=v.shape[-1],
        BLOCK_N=triton.next_power_of_2(kv_group_num),
        num_warps=num_warps,
        num_stages=num_stages
    )
    return
