import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    Prob,  # [batch, head, seq_len, seq_len]
    V,     # [batch, head, seq_len, d_model]
    Req_to_tokens,  # [batch, max_seq_len]
    Out,   # [batch, head, seq_len, d_model]
    B_req_idx,  # [batch]
    B_Start_Loc,  # [batch]
    B_Seqlen,  # [batch]
    stride_prob_b, stride_prob_h, stride_prob_n, stride_prob_m,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_out_b, stride_out_h, stride_out_n, stride_out_d,
    stride_req_to_tokens_b, stride_req_to_tokens_n,
    stride_b_req_idx_b,
    stride_b_start_loc_b,
    stride_b_seqlen_b,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # Compute offsets for the current batch and head
    prob_offset = cur_batch * stride_prob_b + cur_head * stride_prob_h
    v_offset = cur_batch * stride_v_b + cur_head * stride_v_h
    out_offset = cur_batch * stride_out_b + cur_head * stride_out_h

    # Fetch sequence length for the current batch
    seq_len = tl.load(B_Seqlen + cur_batch * stride_b_seqlen_b)

    # Iterate over blocks of tokens
    for block_n in range(0, seq_len, BLOCK_N):
        # Compute the start and end indices for the current block
        start_n = block_n
        end_n = min(block_n + BLOCK_N, seq_len)

        # Initialize the accumulator
        acc = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)

        # Iterate over tokens in the current block
        for n in range(start_n, end_n):
            # Compute the offset for the current token
            token_offset = n * stride_prob_n

            # Fetch the attention probabilities for the current token
            prob = tl.load(Prob + prob_offset + token_offset, mask=n < seq_len, other=0.0)

            # Fetch the value vectors for the current token
            v_offset_n = n * stride_v_n
            v = tl.load(V + v_offset + v_offset_n, mask=n < seq_len, other=0.0)

            # Compute the weighted sum
            acc += prob * v

        # Store the result in the output tensor
        for n in range(start_n, end_n):
            out_offset_n = n * stride_out_n
            tl.store(Out + out_offset + out_offset_n, acc, mask=n < seq_len)

import torch

def token_att_fwd2(
    Prob,  # [batch, head, seq_len, seq_len]
    V,     # [batch, head, seq_len, d_model]
    Req_to_tokens,  # [batch, max_seq_len]
    Out,   # [batch, head, seq_len, d_model]
    B_req_idx,  # [batch]
    B_Start_Loc,  # [batch]
    B_Seqlen,  # [batch]
    BLOCK_DMODEL: int,
    BLOCK_N: int,
    num_warps: int = 4,
    num_stages: int = 3
):
    # Get the batch and head dimensions
    batch, head, seq_len, d_model = Prob.shape

    # Compute the number of key-value groups
    kv_group_num = 1  # Assuming a single key-value group for simplicity

    # Set up the grid size
    grid = (batch, head)

    # Launch the kernel
    _fwd_kernel_token_att2[grid](
        Prob, V, Req_to_tokens, Out,
        B_req_idx, B_Start_Loc, B_Seqlen,
        Prob.stride(0), Prob.stride(1), Prob.stride(2), Prob.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        Req_to_tokens.stride(0), Req_to_tokens.stride(1),
        B_req_idx.stride(0),
        B_Start_Loc.stride(0),
        B_Seqlen.stride(0),
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages
    )
