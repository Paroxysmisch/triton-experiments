import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_token_att2(
    Prob,
    V,
    Out,
    Req_to_tokens,
    stride_prob_bs,
    stride_prob_h,
    stride_v_bs,
    stride_v_h,
    stride_out_bs,
    stride_out_h,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)

    # Calculate the current request's starting index and sequence length
    cur_req_start_index = tl.load(Req_to_tokens + cur_batch)
    cur_req_seq_len = tl.load(Req_to_tokens + cur_batch + 1) - cur_req_start_index

    # Initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Initialize pointers for Prob, V, and Out
    prob_ptrs = Prob + (cur_req_start_index + offs_n[None, :]) * stride_prob_bs + cur_head * stride_prob_h
    v_ptrs = V + (cur_req_start_index + offs_n[:, None]) * stride_v_bs + cur_head * stride_v_h
    out_ptrs = Out + (cur_req_start_index + offs_n[:, None]) * stride_out_bs + cur_head * stride_out_h

    # Load slices of Prob and V
    p_value = tl.load(prob_ptrs, mask=offs_n[None, :] < cur_req_seq_len, other=0.0)
    v_value = tl.load(v_ptrs, mask=offs_n[:, None] < cur_req_seq_len, other=0.0)

    # Initialize accumulator
    acc = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    # Compute attention and accumulate
    acc += tl.dot(p_value, v_value)

    # Store result in Out
    tl.store(out_ptrs, acc, mask=offs_n[:, None] < cur_req_seq_len)

@torch.no_grad()
def token_att_fwd2(prob, v, out, req_to_tokens, max_seq_len):
    BLOCK = 128 if torch.cuda.get_device_capability()[0] >= 8 else 64

    batch, head = prob.shape[0], prob.shape[1]
    grid = (batch, head, triton.cdiv(max_seq_len, BLOCK))
    num_warps = 4 if prob.shape[-1] <= 64 else 8

    _fwd_kernel_token_att2[grid](
        prob,
        v,
        out,
        req_to_tokens,
        prob.stride(0),
        prob.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_N=BLOCK,
        BLOCK_DMODEL=v.shape[-1],
        num_warps=num_warps,
        num_stages=1,
    )
