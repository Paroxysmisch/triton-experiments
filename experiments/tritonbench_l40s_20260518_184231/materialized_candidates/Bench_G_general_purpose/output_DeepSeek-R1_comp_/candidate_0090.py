import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Logics, V, Out,
    B_Loc, B_Start_Loc, B_Seqlen,
    stride_logic_b, stride_logic_h, stride_logic_s,
    stride_v_b, stride_v_h, stride_v_s,
    stride_out_b, stride_out_h, stride_out_d,
    max_len, softmax_scale,
    head_dim,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Get sequence metadata
    start_loc = tl.load(B_Start_Loc + batch_id)
    seq_len = tl.load(B_Seqlen + batch_id)
    
    # Pointers to current batch and head
    logics_ptr = Logics + batch_id * stride_logic_b + head_id * stride_logic_h
    v_ptr = V + batch_id * stride_v_b + head_id * stride_v_h
    out_ptr = Out + batch_id * stride_out_b + head_id * stride_out_h

    # Initialize accumulation variables
    e_max = tl.full((BLOCK_DMODEL,), float('-inf'), dtype=tl.float32)
    e_sum = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)

    # Loop over sequence blocks
    for start_n in range(0, seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Mask to handle variable length sequences
        mask_n = offs_n < seq_len
        current_len = tl.minimum(BLOCK_N, seq_len - start_n)

        # Load logits for current block
        l_ptr = logics_ptr + offs_n
        logits = tl.load(l_ptr, mask=mask_n, other=float('-inf')) * softmax_scale
        
        # Compute local max and update global max
        row_max = tl.max(tl.where(mask_n, logits, float('-inf')), axis=0)
        new_e_max = tl.maximum(e_max, row_max)
        
        # Compute scaled exponentials
        exp_logits = tl.exp(logits - new_e_max)
        exp_logits = tl.where(mask_n, exp_logits, 0.0)
        
        # Update accumulators with new max
        old_scale = tl.exp(e_max - new_e_max)
        acc *= old_scale
        e_sum = e_sum * old_scale + tl.sum(exp_logits, axis=0)
        e_max = new_e_max
        
        # Load values and compute probabilities
        v_offs = (start_loc + start_n + offs_n) * stride_v_s
        v_ptrs = v_ptr + tl.arange(0, BLOCK_DMODEL)[None, :] + v_offs[:, None]
        vs = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        p = exp_logits[:, None] / e_sum
        acc += tl.sum(vs * p, axis=0)

    # Normalize and store results
    acc = acc / e_sum
    tl.store(out_ptr + tl.arange(0, BLOCK_DMODEL), acc)

def token_softmax_reducev_fwd(logics, v, out, B_Loc, B_Start_Loc, B_Seqlen, max_len, softmax_scale):
    BLOCK_DMODEL = logics.shape[-1]
    assert BLOCK_DMODEL == v.shape[-1], "Head dimension mismatch"
    
    batch, n_heads = logics.shape[0], logics.shape[1]
    grid = (batch, n_heads)
    
    _fwd_kernel[grid](
        logics, v, out,
        B_Loc, B_Start_Loc, B_Seqlen,
        logics.stride(0), logics.stride(1), logics.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        max_len, softmax_scale,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=128,
        num_warps=4,
        num_stages=1
    )
