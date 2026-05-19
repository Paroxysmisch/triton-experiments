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
    # Get current batch and head indices
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    
    # Create offset arrays for block processing
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load sequence length info for current batch
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_index = max_input_len - cur_batch_seq_len
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    
    # Calculate base offsets for tensors
    v_loc_off = cur_batch * stride_b_loc_b + (cur_batch_start_index + offs_n) * stride_b_loc_s
    p_offs = cur_head * stride_ph + (cur_batch_in_all_start_index + offs_n) * stride_pbs
    v_offs = cur_head * stride_vh + offs_d[None, :] * stride_vd
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    # Process sequence in blocks
    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load probability and value tensors
        p_value = tl.load(Prob + p_offs + start_n * stride_b_loc_s, 
                         mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0)
        v_loc = tl.load(B_Loc + v_loc_off + start_n * stride_b_loc_s, 
                       mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0)
        v_value = tl.load(V + v_offs + v_loc[:, None] * stride_vbs, 
                         mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)
        
        # Compute weighted sum
        acc += tl.sum(p_value[:, None] * v_value, 0)
    
    # Store result
    acc = acc.to(tl.float16)
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    tl.store(Out + off_o, acc)

@torch.no_grad()
def token_att_fwd2(prob, v, out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len):
    # Determine block size based on Triton version
    BLOCK = 128 if triton.__version__ >= "2.1.0" else 64
    
    # Get dimensions
    batch, head = B_Loc.shape[0], v.shape[1]
    dim = v.shape[-1]
    
    # Launch kernel
    grid = (batch, head)
    _fwd_kernel_token_att2[grid](
        prob, v, out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
        B_Loc.stride(0), B_Loc.stride(1),
        prob.stride(0), prob.stride(1),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_DMODEL=dim,
        BLOCK_N=BLOCK,
        num_warps=4,
        num_stages=1,
    )
