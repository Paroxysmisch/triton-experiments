import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att1(
    Q, 
    K, 
    B_Loc, 
    B_Start_Loc, 
    B_Seqlen, 
    Att_Out,
    stride_qbs, 
    stride_qh,
    stride_kbs, 
    stride_kh,
    stride_att_bs, 
    stride_att_h,
    max_input_len, 
    sm_scale,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
):
    # program IDs
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    block_idx = tl.program_id(2)

    # Starting index for this block in the sequence dimension
    start_n = block_idx * BLOCK_N

    # Load sequence length and start index for the current batch
    seq_len = tl.load(B_Seqlen + batch_id)
    seq_start = tl.load(B_Start_Loc + batch_id)

    # Offsets in the Q/K matrices
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Global offset for Q and K
    # Q shape: (total_seq, num_heads, d_model)
    # K shape: (total_seq, num_heads, d_model)
    q_offset = (
        (seq_start + offs_m[:, None]) * stride_qbs
        + head_id * stride_qh
    )
    k_offset = (
        (seq_start + start_n + offs_n[None, :]) * stride_kbs
        + head_id * stride_kh
    )

    # Mask computation to ensure we don't read/write out of bounds
    q_mask = offs_m < seq_len
    k_mask = (start_n + offs_n) < seq_len

    # Load Q and K blocks
    q_block = tl.load(Q + q_offset, mask=q_mask[:, None], other=0.0)
    k_block = tl.load(K + k_offset, mask=k_mask[None, :], other=0.0)

    # Perform dot product along d_model dimension
    # Q and K must have the same feature dimension, implicitly deduce from strides
    att_scores = tl.dot(q_block, tl.trans(k_block))

    # Scale the scores
    att_scores = att_scores * sm_scale

    # Store attention scores in Att_Out
    # Att_Out shape: (total_seq, num_heads, total_seq)
    # Offsets for writing
    att_out_offset = (
        (seq_start + offs_m[:, None]) * stride_att_bs
        + head_id * stride_att_h
        + (seq_start + start_n + offs_n[None, :])
    )
    # Use the same mask for storing results
    out_mask = (q_mask[:, None] & k_mask[None, :])
    tl.store(Att_Out + att_out_offset, att_scores, mask=out_mask)

def token_att_fwd(Q, K, B_Loc, B_Start_Loc, B_Seqlen, Att_Out, max_input_len):
    # Infer shapes
    d_model = Q.shape[-1]
    sm_scale = 1.0 / (d_model**0.5)
    batch = B_Seqlen.shape[0]
    heads = Q.shape[1]

    # Decide block sizes and warps
    if triton.compiletime.get_current_device_compute_capability()[0] >= 8:
        BLOCK = 128
        num_warps = 4
    else:
        BLOCK = 64
        num_warps = 2

    # Grid: (batch, heads, ceil_div(max_input_len, BLOCK))
    grid = (
        batch,
        heads,
        (max_input_len + BLOCK - 1) // BLOCK,
    )

    _fwd_kernel_token_att1[grid](
        Q,
        K,
        B_Loc,
        B_Start_Loc,
        B_Seqlen,
        Att_Out,
        Q.stride(0),
        Q.stride(1),
        K.stride(0),
        K.stride(1),
        Att_Out.stride(0),
        Att_Out.stride(1),
        max_input_len,
        sm_scale,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
