import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    Prob,
    V,
    Out,
    Req_to_tokens,
    B_Start_Loc,
    B_Seqlen,
    stride_prob_t, stride_prob_h, stride_prob_k,
    stride_v_t, stride_v_h, stride_v_d,
    stride_out_t, stride_out_h, stride_out_d,
    stride_req_to_tokens_b, stride_req_to_tokens_t,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

    # Offsets for the current block of queries
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load token indices for queries in the current block
    req_to_tokens_offsets_m = cur_batch * stride_req_to_tokens_b + offs_m * stride_req_to_tokens_t
    token_indices_m = tl.load(Req_to_tokens + req_to_tokens_offsets_m, mask=offs_m < cur_batch_seq_len, other=0)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    # Loop over key blocks
    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        # Load token indices for keys in the current block
        req_to_tokens_offsets_n = cur_batch * stride_req_to_tokens_b + offs_n * stride_req_to_tokens_t
        token_indices_n = tl.load(Req_to_tokens + req_to_tokens_offsets_n, mask=offs_n < cur_batch_seq_len, other=0)

        # Load probabilities for current query and key block
        prob_offsets = token_indices_m[:, None] * stride_prob_t + cur_head * stride_prob_h + token_indices_n[None, :] * stride_prob_k
        p = tl.load(Prob + prob_offsets, 
                    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_n[None, :] < cur_batch_seq_len), 
                    other=0.0)

        # Load values for current key block
        v_offsets = token_indices_n[:, None] * stride_v_t + cur_kv_head * stride_v_h + offs_d[None, :] * stride_v_d
        v = tl.load(V + v_offsets, 
                    mask=offs_n[:, None] < cur_batch_seq_len, 
                    other=0.0)

        # Compute acc += p * v
        p = p.to(v.dtype)
        acc += tl.dot(p, v)

    # Store the accumulated result
    out_offsets = token_indices_m[:, None] * stride_out_t + cur_head * stride_out_h + offs_d[None, :] * stride_out_d
    tl.store(Out + out_offsets, acc, mask=offs_m[:, None] < cur_batch_seq_len)

@torch.no_grad()
def token_att_fwd2(Prob, V, Out, Req_to_tokens, B_Start_Loc, B_Seqlen):
    if torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability()
        if device_capability[0] >= 8:
            BLOCK = 128
        else:
            BLOCK = 64
    else:
        BLOCK = 64  # Default if CUDA not available

    D_MODEL = V.size(-1)
    assert D_MODEL in {16, 32, 64, 128, 256}, "D_MODEL must be one of 16, 32, 64, 128, 256"

    batch_size, num_heads = B_Seqlen.size(0), Prob.size(1)
    kv_group_num = num_heads // V.size(1)

    max_seq_len = torch.max(B_Seqlen).item()
    num_blocks = triton.cdiv(max_seq_len, BLOCK)

    grid = (batch_size, num_heads, num_blocks)
    num_warps = 4 if D_MODEL <= 64 else 8

    _fwd_kernel_token_att2[grid](
        Prob, V, Out,
        Req_to_tokens, B_Start_Loc, B_Seqlen,
        Prob.stride(0), Prob.stride(1), Prob.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        Req_to_tokens.stride(0), Req_to_tokens.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=D_MODEL,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1
    )
