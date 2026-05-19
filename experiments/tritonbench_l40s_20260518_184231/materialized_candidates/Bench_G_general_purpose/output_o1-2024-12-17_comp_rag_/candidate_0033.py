import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_token_att2(
    Prob,           # [batch, head, tokens]
    V,              # [batch, head, tokens, dmodel]
    Out,            # [batch, head, tokens, dmodel]
    Req_to_tokens,  # [batch] -> mapping from request idx to first token
    B_Seqlen,       # [batch] -> number of tokens for each batch
    stride_prob_b, stride_prob_h, stride_prob_t,
    stride_v_b, stride_v_h, stride_v_t,
    stride_out_b, stride_out_h,
    BLOCK_M: tl.constexpr,      # tokens per block
    BLOCK_DMODEL: tl.constexpr, # model dimension
    BLOCK_N: tl.constexpr       # tokens per block
):
    # program_id axes: 0 -> batch, 1 -> head, 2 -> block idx
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    block_idx = tl.program_id(2)

    # load sequence length for the current batch
    seqlen = tl.load(B_Seqlen + cur_batch)
    # find the token offset for the current batch
    batch_tok_offset = tl.load(Req_to_tokens + cur_batch)

    # each block processes BLOCK_M tokens (dimension M)
    offs_m = block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    # clamp valid tokens
    mask_m = offs_m < seqlen

    # prepare pointers for Prob and V
    # Prob is [batch, head, tokens], so offset is
    # (cur_batch * stride_prob_b) + (cur_head * stride_prob_h) + (offs_m * stride_prob_t)
    prob_ptrs = Prob + (
        cur_batch * stride_prob_b
        + cur_head * stride_prob_h
        + (batch_tok_offset + offs_m) * stride_prob_t
    )

    # accumulation buffer for final result: [BLOCK_M, BLOCK_DMODEL]
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    # iterate over the tokens in chunks of BLOCK_N
    # each iteration loads V chunk and the corresponding Prob chunk
    # multiply and accumulate in acc
    for start_n in range(0, seqlen, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seqlen

        # load prob slice: shape [BLOCK_M, BLOCK_N]
        p = tl.load(prob_ptrs + offs_n, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

        # V is [batch, head, tokens, dmodel]
        # offset for V: (cur_batch * stride_v_b) + (cur_head * stride_v_h) + (offs_n * stride_v_t)
        v_ptrs = V + (
            cur_batch * stride_v_b
            + cur_head * stride_v_h
            + (batch_tok_offset + offs_n) * stride_v_t
        )
        # shape: [BLOCK_N, BLOCK_DMODEL]
        v_data = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # accumulate
        # p shape: [BLOCK_M, BLOCK_N], v_data shape: [BLOCK_N, BLOCK_DMODEL]
        acc += tl.dot(p, v_data)

    # store results to Out
    out_ptrs = Out + (
        cur_batch * stride_out_b
        + cur_head * stride_out_h
        + (batch_tok_offset + offs_m)[:, None] * BLOCK_DMODEL
    )

    # write back
    # mask is [BLOCK_M, BLOCK_DMODEL] with dimension M valid check
    tl.store(out_ptrs, acc, mask=mask_m[:, None])


@torch.no_grad()
def token_att_fwd2(prob, v, out, req_to_tokens, b_seqlen):
    """
    prob: [batch, head, tokens]
    v:    [batch, head, tokens, dmodel]
    out:  [batch, head, tokens, dmodel]
    req_to_tokens: [batch], base offset of tokens for each request
    b_seqlen:      [batch], number of tokens in each request
    """
    # device capability based BLOCK setting example
    cap_major = torch.cuda.get_device_capability()
    if cap_major[0] >= 8:
        BLOCK = 128
    else:
        BLOCK = 64

    batch = prob.shape[0]
    head = prob.shape[1]
    max_tokens = prob.shape[2]  # maximum tokens dimension
    dmodel = v.shape[3]

    grid = (batch, head, triton.cdiv(max_tokens, BLOCK))
    num_warps = 4 if dmodel <= 64 else 8

    _fwd_kernel_token_att2[grid](
        prob, v, out,
        req_to_tokens,
        b_seqlen,
        prob.stride(0), prob.stride(1), prob.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=dmodel,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
