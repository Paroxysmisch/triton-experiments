import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    Prob, V, Out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
    stride_b_loc_b, stride_b_loc_s,
    stride_ph, stride_pbs,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # Determine batch and head indices
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    
    # Create ranges for block processing
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load sequence metadata
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_index = max_input_len - cur_batch_seq_len
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    # Initialize accumulator
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    # Process each token block
    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Calculate probability offsets
        p_offs = cur_head * stride_ph + (cur_batch_in_all_start_index + start_n + offs_n) * stride_pbs
        p_mask = (start_n + offs_n) < cur_batch_seq_len
        p_value = tl.load(Prob + p_offs, mask=p_mask, other=0.0)

        # Calculate value locations
        loc_offs = cur_batch * stride_b_loc_b + (cur_batch_start_index + start_n + offs_n) * stride_b_loc_s
        v_loc = tl.load(B_Loc + loc_offs, mask=p_mask, other=0.0)

        # Calculate value offsets and load values
        v_offs = v_loc[:, None] * stride_vbs + cur_head * stride_vh + offs_d[None, :] * stride_vd
        v_value = tl.load(V + v_offs, mask=p_mask[:, None], other=0.0)

        # Accumulate weighted values
        acc += tl.sum(p_value[:, None] * v_value, axis=0)

    # Store result
    out_offs = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    tl.store(Out + out_offs, acc.to(tl.float16))

@torch.no_grad()
def token_att_fwd2(prob, v, out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len):
    # Configure kernel parameters
    BLOCK = 128 if triton.__version__ >= "2.1.0" else 64
    batch, head = B_Loc.shape[0], v.shape[1]
    grid = (batch, head)
    
    # Launch kernel
    _fwd_kernel_token_att2[grid](
        prob, v, out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
        B_Loc.stride(0), B_Loc.stride(1),
        prob.stride(0), prob.stride(1),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_DMODEL=v.shape[-1],
        BLOCK_N=BLOCK,
        num_warps=4,
        num_stages=1
    )
