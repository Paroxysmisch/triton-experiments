import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_token_att2(
    Prob, V, Out,
    Req_to_tokens,
    stride_pbs, stride_ph,
    stride_vbs, stride_vh,
    stride_obs, stride_oh,
    BLOCK_N: tl.constexpr,
):
    # Program ID gives current batch and head
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    
    # Get sequence length for current batch from Req_to_tokens
    cur_seq_len = tl.load(Req_to_tokens + cur_batch)
    
    # Initialize offset calculations
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_N)  # Using same size for simplicity
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    # Calculate base offsets for tensors
    prob_offset = cur_batch * stride_pbs + cur_head * stride_ph
    v_offset = cur_batch * stride_vbs + cur_head * stride_vh
    
    # Process tokens in blocks
    for start_n in range(0, cur_seq_len, BLOCK_N):
        # Load probability values
        p_offs = prob_offset + (start_n + offs_n)
        p_mask = (start_n + offs_n) < cur_seq_len
        p_value = tl.load(Prob + p_offs, mask=p_mask, other=0.0)
        
        # Load value tensor
        v_offs = v_offset + (start_n + offs_n)
        v_value = tl.load(V + v_offs, mask=p_mask, other=0.0)
        
        # Multiply and accumulate
        acc += p_value * v_value
    
    # Store result in output tensor
    out_offset = cur_batch * stride_obs + cur_head * stride_oh
    out_ptr = Out + out_offset
    tl.store(out_ptr, acc.to(Out.dtype.element_ty))

@torch.no_grad()
def token_att_fwd2(prob, v, req_to_tokens):
    # Determine block size based on GPU capability
    BLOCK = 128 if torch.cuda.get_device_capability()[0] >= 8 else 64
    
    # Get dimensions
    batch_size = req_to_tokens.shape[0]
    num_heads = prob.shape[1]
    kv_group_num = prob.shape[1] // v.shape[1]
    
    # Initialize output tensor
    output = torch.empty_like(v)
    
    # Calculate grid dimensions
    grid = (batch_size, num_heads)
    
    # Launch kernel
    _fwd_kernel_token_att2[grid](
        prob, v, output,
        req_to_tokens,
        prob.stride(0), prob.stride(1),
        v.stride(0), v.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_N=BLOCK,
        num_warps=4,
        num_stages=1
    )
    
    return output
