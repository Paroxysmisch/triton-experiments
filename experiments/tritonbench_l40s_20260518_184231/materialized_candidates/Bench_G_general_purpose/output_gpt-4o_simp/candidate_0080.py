import triton
import triton.language as tl

# Triton kernel for attention-like operation
@triton.jit
def _fwd_kernel_token_att2(Prob_ptr, V_ptr, Out_ptr, 
                           Batch, Heads, Seq_len, 
                           Block_size_m, Block_size_n, 
                           stride_prob_m, stride_prob_h, stride_prob_s, 
                           stride_v_m, stride_v_h, stride_v_s, 
                           stride_out_m, stride_out_h, stride_out_s,
                           **meta):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate the start index of each block
    start_m = pid_m * Block_size_m
    start_n = pid_n * Block_size_n
    
    # Create block indices
    offsets_m = start_m + tl.arange(0, Block_size_m)
    offsets_n = start_n + tl.arange(0, Block_size_n)
    
    # Initialize output accumulator
    Out_acc = tl.zeros([Block_size_m, Block_size_n], dtype=tl.float32)
    
    # Iterate over the sequence
    for seq_idx in range(0, Seq_len, Block_size_n):
        # Load probabilities and values for the current block
        Prob = tl.load(Prob_ptr + offsets_m[:, None] * stride_prob_m + 
                       offsets_n[None, :] * stride_prob_s + 
                       seq_idx * stride_prob_s, 
                       mask=offsets_m[:, None] < Batch * Heads * Seq_len)
        
        V = tl.load(V_ptr + offsets_m[:, None] * stride_v_m + 
                    offsets_n[None, :] * stride_v_s + 
                    seq_idx * stride_v_s, 
                    mask=offsets_m[:, None] < Batch * Heads * Seq_len)
        
        # Compute weighted sum
        Out_acc += Prob @ V
    
    # Store the result
    tl.store(Out_ptr + offsets_m[:, None] * stride_out_m + 
             offsets_n[None, :] * stride_out_s, Out_acc)

# Wrapper function to launch the Triton kernel
def token_att_fwd2(Prob, V, Out, Batch, Heads, Seq_len, 
                   Block_size_m=128, Block_size_n=128):
    assert Prob.shape == (Batch, Heads, Seq_len, Seq_len)
    assert V.shape == (Batch, Heads, Seq_len, -1)
    assert Out.shape == (Batch, Heads, Seq_len, -1)
    
    # Strides for input tensors
    stride_prob_m = Prob.stride(0)
    stride_prob_h = Prob.stride(1)
    stride_prob_s = Prob.stride(2)
    
    stride_v_m = V.stride(0)
    stride_v_h = V.stride(1)
    stride_v_s = V.stride(2)
    
    stride_out_m = Out.stride(0)
    stride_out_h = Out.stride(1)
    stride_out_s = Out.stride(2)
    
    # Launch kernel
    grid = (triton.cdiv(Batch * Heads, Block_size_m), triton.cdiv(Seq_len, Block_size_n))
    _fwd_kernel_token_att2[grid](
        Prob, V, Out,
        Batch, Heads, Seq_len,
        Block_size_m, Block_size_n,
        stride_prob_m, stride_prob_h, stride_prob_s,
        stride_v_m, stride_v_h, stride_v_s,
        stride_out_m, stride_out_h, stride_out_s
    )
