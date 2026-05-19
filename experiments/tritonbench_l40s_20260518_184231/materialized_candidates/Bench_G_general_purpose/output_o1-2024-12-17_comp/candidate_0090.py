import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Logits_ptr, V_ptr, Out_ptr,
    B_Loc_ptr, B_Start_Loc_ptr, B_Seqlen_ptr,
    stride_log_b, stride_log_h, stride_log_n,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_out_b, stride_out_h, stride_out_n, stride_out_d,
    BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)

    # Load sequence info
    b_start = tl.load(B_Start_Loc_ptr + b_id)
    seqlen = tl.load(B_Seqlen_ptr + b_id)

    # Base offsets
    logits_b_offset = b_id * stride_log_b
    logits_h_offset = h_id * stride_log_h
    v_b_offset = b_id * stride_v_b
    v_h_offset = h_id * stride_v_h
    out_b_offset = b_id * stride_out_b
    out_h_offset = h_id * stride_out_h

    # Initialize accumulators
    e_max = tl.float32(-1e9)
    e_sum = tl.float32(0)
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_N)
    start_loc = tl.load(B_Loc_ptr + b_id)
    n_blocks = (seqlen + BLOCK_N - 1) // BLOCK_N

    # --- Compute e_max ---
    for block_idx in range(n_blocks):
        n_offset = block_idx * BLOCK_N + offs_n
        mask = n_offset < seqlen
        loc_offset = start_loc + n_offset
        logits_offset = logits_b_offset + logits_h_offset + loc_offset * stride_log_n
        logits_val = tl.where(
            mask,
            tl.load(Logits_ptr + logits_offset, mask=mask, other=0.0),
            -1e9
        )
        block_max = tl.maximum(tl.max(logits_val, 0), e_max)
        e_max = tl.maximum(block_max, e_max)

    # --- Compute e_sum and accum ---
    for block_idx in range(n_blocks):
        n_offset = block_idx * BLOCK_N + offs_n
        mask = n_offset < seqlen
        loc_offset = start_loc + n_offset
        logits_offset = logits_b_offset + logits_h_offset + loc_offset * stride_log_n
        v_offset = v_b_offset + v_h_offset + loc_offset * stride_v_n

        logits_val = tl.where(
            mask,
            tl.load(Logits_ptr + logits_offset, mask=mask, other=0.0),
            -1e9
        )
        log_shifted = logits_val - e_max
        exps = tl.exp(log_shifted)
        e_sum += tl.sum(exps, 0)

        # Load V and accumulate
        v_vec = tl.load(
            V_ptr + v_offset[:, None] + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_v_d,
            mask=mask[:, None],
            other=0.0
        )
        v_contrib = exps[:, None] * v_vec
        acc += tl.sum(v_contrib, 0)

    # Normalize and store
    inv_sum = 1.0 / e_sum
    out_offset = out_b_offset + out_h_offset + b_start * stride_out_n
    out_ptrs = Out_ptr + out_offset * 0  # offset for block of DMODEL
    out_vals = acc * inv_sum

    # Write result
    # Because sequences can be variable-length, each token's out might need separate indexing
    # We'll do a single write here for the final "summation" vector if the reduction implies that shape
    offs_d = tl.arange(0, BLOCK_DMODEL)
    tl.store(out_ptrs + offs_d * stride_out_d, out_vals)


def token_softmax_reducev_fwd(
    Logits, V, Out,
    B_Loc, B_Start_Loc, B_Seqlen,
    BLOCK_N, BLOCK_DMODEL,
    num_warps=4, num_stages=2
):
    batch = B_Loc.shape[0]
    # Assume Logits shape: [batch, heads, max_seqlen]
    # V shape: [batch, heads, max_seqlen, d_model]
    # Out shape: [batch, heads, ..., d_model] (depending on usage)
    heads = Logits.shape[1]

    grid = (batch, heads)
    triton.run(
        _fwd_kernel,
        grid=grid,
        num_warps=num_warps,
        num_stages=num_stages,
        args=[
            Logits, V, Out,
            B_Loc, B_Start_Loc, B_Seqlen,
            Logits.stride(0), Logits.stride(1), Logits.stride(2),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
            BLOCK_N, BLOCK_DMODEL
        ]
    )
