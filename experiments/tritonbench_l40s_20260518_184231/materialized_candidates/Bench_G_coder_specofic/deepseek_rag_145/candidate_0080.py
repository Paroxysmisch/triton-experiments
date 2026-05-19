import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    Prob, V, Req_to_tokens, Out, B_req_idx, B_Start_Loc, B_Seqlen, 
    stride_p, stride_v, stride_r, stride_o,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_index = max_input_len - cur_batch_seq_len
    cur_batch_end_index = cur_batch_seq_len
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    p_offs = cur_head * stride_ph + (cur_batch_in_all_start_index + offs_n) * stride_pbs
    v_offs = cur_head * stride_vh + offs_d[None, :] * stride_vd
    r_offs = cur_head * stride_rh + offs_n * stride_rs

    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        p_value = tl.load(Prob + p_offs + start_n * stride_p, mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0)
        v_value = tl.load(V + v_offs + offs_d[:, None] * stride_v, mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)
        req_idx = tl.load(Req_to_tokens + r_offs + start_n * stride_r, mask=(start_n + offs_n) < cur_batch_seq_len, other=0)
        v_value = tl.gather(v_value, req_idx, 0)
        acc += tl.sum(p_value[:, None] * v_value, 0)

    acc = acc.to(tl.float16)
    off_o = cur_batch * stride_ob + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)

@torch.no_grad()
def token_att_fwd2(prob, v, Req_to_tokens, out, B_req_idx, B_Start_Loc, B_Seqlen, max_input_len):
    if triton.__version__ >= "2.1.0":
        BLOCK = 128
    else:
        BLOCK = 64
    batch, head = B_Start_Loc.shape[0], v.shape[1]
    grid = (batch, head)
    num_warps = 4
    dim = v.shape[-1]

    # Determine the total number of sequences and calculate the sequence block size
    BLOCK_seq = BLOCK
    num_seq = (max_input_len + BLOCK_seq - 1) // BLOCK_seq

    _fwd_kernel_token_att2[grid](
        prob, v, Req_to_tokens, out, B_req_idx, B_Start_Loc, B_Seqlen, 
        prob.stride(0), v.stride(0), Req_to_tokens.stride(0), out.stride(0),
        BLOCK_DMODEL=dim,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=num_seq,
    )
