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
    # Get current batch and head indices from program ID
    pid_batch = tl.program_id(0)  # batch dimension
    pid_head = tl.program_id(1)   # head dimension
    
    # Create offset array for the block
    offs = tl.arange(0, BLOCK_SIZE)
    
    # Load sequence length and start position for current batch
    seq_len = tl.load(B_Seqlen + pid_batch)
    batch_start = tl.load(B_Start_Loc + pid_batch)
    
    # Calculate global offsets and create mask for valid elements
    global_offs = batch_start + offs
    mask = offs < seq_len
    
    # Load logits for current head and batch
    # Use mask to handle padding and set masked values to -inf
    logits = tl.load(
        Logics + pid_head * stride_logic_h + global_offs * stride_logic_bs,
        mask=mask, other=-float('inf')
    ).to(tl.float32)
    
    # Compute softmax in a numerically stable way:
    # 1. Subtract max for numerical stability
    max_logits = tl.max(logits, axis=0)
    logits_shifted = logits - max_logits
    
    # 2. Compute exponentials
    numerator = tl.exp(logits_shifted)
    
    # 3. Compute sum for normalization
    denominator = tl.sum(numerator, axis=0)
    
    # 4. Normalize to get probabilities
    probs = numerator / denominator
    
    # Store the results
    tl.store(
        Prob_Out + pid_head * stride_prob_h + global_offs * stride_prob_bs,
        probs, mask=mask
    )

@torch.no_grad()
def token_softmax_fwd(Logics, B_Start_Loc, B_Seqlen, Prob_Out, max_input_len):
    # Calculate block size as next power of 2 of max_input_len
    BLOCK_SIZE = triton.next_power_of_2(max_input_len)
    
    # Get batch size and number of heads
    batch_size = B_Start_Loc.shape[0]
    num_heads = Logics.shape[0]
    
    # Determine optimal number of warps based on block size
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    
    # Launch kernel with grid dimensions (batch_size, num_heads)
    _fwd_kernel_token_softmax[(batch_size, num_heads)](
        Logics, B_Start_Loc, B_Seqlen, Prob_Out,
        Logics.stride(0), Logics.stride(1),
        Prob_Out.stride(0), Prob_Out.stride(1),
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return
