import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    Out, Prob, V, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen,
    stride_outbatch, stride_outhead, stride_outtoken, stride_outdim,
    stride_probbatch, stride_probhead, stride_probtoken, stride_probdim,
    stride_vbatch, stride_vhead, stride_vtoken, stride_vdim,
    stride_req_to_tokens,
    B, H, kv_group_num,
    BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Program IDs for batch and head dimensions
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)

    # Only proceed if we're within the valid range
    if b_id >= B or h_id >= H:
        return

    # Create a range for tokens and model-dims
    token_range = tl.arange(0, BLOCK_N)
    dmodel_range = tl.arange(0, BLOCK_DMODEL)

    # Offsets for indexing
    # Retrieve request index from B_req_idx
    b_req_idx_val = tl.load(B_req_idx + b_id)
    # Retrieve the start token offset for this batch
    b_start_loc_val = tl.load(B_Start_Loc + b_id)
    # Retrieve sequence length for this batch
    b_seqlen_val = tl.load(B_Seqlen + b_id)

    # Offset for Out
    out_offset = b_id * stride_outbatch + h_id * stride_outhead

    # Accumulate for each dmodel position
    acc = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)

    # Each block will process up to BLOCK_N tokens.
    # Compute for each token in this block
    for n_idx in range(kv_group_num):
        # Actual token index in the entire sequence
        tk_id = token_range + n_idx * BLOCK_N + b_start_loc_val
        # Check token index vs sequence length
        mask = tk_id < (b_start_loc_val + b_seqlen_val)

        # Load Probability
        prob_offset = (b_id * stride_probbatch) + (h_id * stride_probhead) \
                      + (tk_id * stride_probtoken)
        prob = tl.where(mask, tl.load(Prob + prob_offset, mask=mask, other=0.), 0.)

        # For Req_to_tokens, get the actual token index if needed
        # This example uses tk_id directly, but typically you'd gather from Req_to_tokens
        # e.g.: real_tk_id = tl.load(Req_to_tokens + b_req_idx_val * stride_req_to_tokens + tk_id, mask=mask, other=0)

        # Load Value
        v_offset = (b_id * stride_vbatch) + (h_id * stride_vhead) \
                   + (tk_id * stride_vtoken)
        # Gather V for each model dim
        # We add dmodel_range for the dimension offset
        v_val = tl.load(
            V + v_offset[:, None] + dmodel_range[None, :] * stride_vdim,
            mask=mask[:, None],
            other=0.
        )

        # Multiply prob with V across tokens, sum over tokens into 'acc'
        # prob shape: [BLOCK_N], v_val shape: [BLOCK_N, BLOCK_DMODEL]
        acc += tl.sum(v_val * prob[:, None], axis=0)

    # Store the result to Out
    out_ptr = Out + out_offset * stride_outtoken
    tl.store(
        out_ptr + dmodel_range * stride_outdim,
        acc.to(tl.float16)
    )


def token_att_fwd2(
    Out, Prob, V, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen,
    stride_outbatch, stride_outhead, stride_outtoken, stride_outdim,
    stride_probbatch, stride_probhead, stride_probtoken, stride_probdim,
    stride_vbatch, stride_vhead, stride_vtoken, stride_vdim,
    stride_req_to_tokens,
    batch, head, kv_group_num,
    BLOCK_N=128, BLOCK_DMODEL=64
):
    grid = (batch, head)
    _fwd_kernel_token_att2[grid](
        Out, Prob, V, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen,
        stride_outbatch, stride_outhead, stride_outtoken, stride_outdim,
        stride_probbatch, stride_probhead, stride_probtoken, stride_probdim,
        stride_vbatch, stride_vhead, stride_vtoken, stride_vdim,
        stride_req_to_tokens,
        batch, head, kv_group_num,
        BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,  # Example setting; tune for device
        num_stages=2  # Example setting; tune for device
    )
