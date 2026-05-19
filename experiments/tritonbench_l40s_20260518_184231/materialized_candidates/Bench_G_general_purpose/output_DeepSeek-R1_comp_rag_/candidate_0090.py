import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Logics, V, Out,
    B_Loc, B_Start_Loc, B_Seqlen,
    stride_logic_b, stride_logic_h, stride_logic_s,
    stride_v_b, stride_v_s, stride_v_h, stride_v_d,
    stride_out_b, stride_out_h, stride_out_d,
    H,  # Number of heads
    D_MODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Extract program IDs for batch and head
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    
    # Load the start location and sequence length for the current batch
    start_loc = tl.load(B_Start_Loc + batch_id)
    seq_len = tl.load(B_Seqlen + batch_id)
    
    # Calculate pointers for the current batch and head
    logic_ptr = Logics + (batch_id * stride_logic_b) + (head_id * stride_logic_h) + (start_loc * stride_logic_s)
    v_ptr = V + (batch_id * stride_v_b) + (head_id * stride_v_h) + (start_loc * stride_v_s)
    out_ptr = Out + (batch_id * stride_out_b) + (head_id * stride_out_h)
    
    # Initialize accumulators
    e_max = tl.full((BLOCK_DMODEL,), -float('inf'), dtype=tl.float32)
    e_sum = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)
    
    # Process each block in the sequence
    for start_n in range(0, seq_len, BLOCK_N):
        # Determine the current block size
        n_remaining = seq_len - start_n
        block_n = min(BLOCK_N, n_remaining)
        
        # Load logits for the current block
        logits = tl.load(logic_ptr + start_n + tl.arange(0, block_n), mask=(tl.arange(0, block_n) < block_n), other=-float('inf'))
        
        # Compute the current max and update the running max
        current_max = tl.max(logits)
        new_max = tl.maximum(e_max, current_max)
        
        # Compute exponents for the current block adjusted by new_max
        logits_minus_max = logits - new_max
        exp_logits = tl.exp(logits_minus_max)
        
        # Sum of exponents for the current block
        block_sum_exp = tl.sum(exp_logits)
        
        # Adjust previous accumulators based on the new_max
        old_scale = tl.exp(e_max - new_max)
        e_sum = e_sum * old_scale + block_sum_exp
        acc = acc * old_scale
        
        # Update the running max
        e_max = new_max
        
        # Load the corresponding V values for the current block
        offs_n = start_n + tl.arange(0, block_n)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        v_ptrs = v_ptr + (offs_n[:, None] * stride_v_s) + (offs_d[None, :] * stride_v_d)
        mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < D_MODEL)
        v = tl.load(v_ptrs, mask=mask, other=0.0)
        
        # Update the accumulator with the current block's contribution
        acc += tl.sum(exp_logits[:, None] * v, axis=0)
    
    # Normalize the accumulated values by the sum of exponents
    acc = acc / e_sum
    
    # Store the result
    offs_d = tl.arange(0, BLOCK_DMODEL)
    tl.store(out_ptr + offs_d * stride_out_d, acc, mask=offs_d < D_MODEL)

def token_softmax_reducev_fwd(logics, v, out, B_Loc, B_Start_Loc, B_Seqlen):
    # Extract dimensions
    batch_size, H = logics.shape[0], logics.shape[1]
    D_MODEL = v.shape[-1]
    
    # Define grid
    grid = (batch_size, H)
    
    # Compute strides
    stride_logic_b = logics.stride(0)
    stride_logic_h = logics.stride(1)
    stride_logic_s = logics.stride(2)
    
    stride_v_b = v.stride(0)
    stride_v_s = v.stride(1)
    stride_v_h = v.stride(2)
    stride_v_d = v.stride(3)
    
    stride_out_b = out.stride(0)
    stride_out_h = out.stride(1)
    stride_out_d = out.stride(2)
    
    # Determine block sizes
    BLOCK_N = 64  # Tune based on performance
    BLOCK_DMODEL = triton.next_power_of_2(D_MODEL)
    
    # Configure kernel parameters
    num_warps = 8
    num_stages = 3
    
    # Launch kernel
    _fwd_kernel[grid](
        logics, v, out,
        B_Loc, B_Start_Loc, B_Seqlen,
        stride_logic_b, stride_logic_h, stride_logic_s,
        stride_v_b, stride_v_s, stride_v_h, stride_v_d,
        stride_out_b, stride_out_h, stride_out_d,
        H,
        D_MODEL,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=num_stages,
    )
