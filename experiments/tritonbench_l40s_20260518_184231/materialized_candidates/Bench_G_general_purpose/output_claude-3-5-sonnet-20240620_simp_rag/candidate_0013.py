import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_softmax(
    Logics, B_Start_Loc, B_Seqlen, Prob_Out,
    stride_logic_h, stride_logic_bs,
    stride_prob_h, stride_prob_bs,
    BLOCK_SIZE: tl.constexpr
):
    # Get current batch and head indices being processed
    pid_batch = tl.program_id(0)  # batch dimension
    pid_head = tl.program_id(1)   # head dimension
    
    # Create offset array for parallel processing
    offs = tl.arange(0, BLOCK_SIZE)
    
    # Load sequence length and start position for current batch
    seq_len = tl.load(B_Seqlen + pid_batch)
    start_idx = tl.load(B_Start_Loc + pid_batch)
    
    # Calculate input pointer for current head and batch
    in_ptr = Logics + pid_head * stride_logic_h + \
             (start_idx + offs) * stride_logic_bs
    
    # Load logits with masking for valid sequence length
    logits = tl.load(in_ptr, mask=offs < seq_len, other=-float('inf'))
    logits = logits.to(tl.float32)
    
    # Compute softmax:
    # 1. Subtract max for numerical stability
    # 2. Compute exp
    # 3. Normalize by sum
    max_logits = tl.max(logits, axis=0)
    logits = logits - max_logits
    numerator = tl.exp(logits)
    denominator = tl.sum(numerator, axis=0)
    probs = numerator / denominator
    
    # Store results
    out_ptr = Prob_Out + pid_head * stride_prob_h + \
              (start_idx + offs) * stride_prob_bs
    tl.store(out_ptr, probs, mask=offs < seq_len)

@torch.no_grad()
def token_softmax_fwd(Logics, B_Start_Loc, B_Seqlen, Prob_Out, max_input_len):
    # Compute block size as next power of 2 of max sequence length
    BLOCK_SIZE = triton.next_power_of_2(max_input_len)
    
    # Get batch size and number of heads
    batch_size = B_Start_Loc.shape[0]
    num_heads = Logics.shape[0]
    
    # Determine optimal number of warps based on block size
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 4096:
        num_warps = 16
    
    # Launch kernel with grid of (batch_size, num_heads)
    _fwd_kernel_token_softmax[(batch_size, num_heads)](
        Logics, B_Start_Loc, B_Seqlen, Prob_Out,
        Logics.stride(0), Logics.stride(1),
        Prob_Out.stride(0), Prob_Out.stride(1),
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE
    )
