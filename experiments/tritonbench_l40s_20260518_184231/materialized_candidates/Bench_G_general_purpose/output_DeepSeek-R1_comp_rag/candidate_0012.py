import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_softmax(
    Logics, B_Start_Loc, B_Seqlen,
    Prob_Out,
    stride_logic_h, stride_logic_bs,
    stride_prob_h, stride_prob_bs,
    BLOCK_SIZE: tl.constexpr
):
    # Determine current batch and head indices
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    
    # Calculate column offsets and load sequence info
    col_offsets = tl.arange(0, BLOCK_SIZE)
    cur_seq_len = tl.load(B_Seqlen + cur_batch)
    start_loc = tl.load(B_Start_Loc + cur_batch)
    
    # Load logits for current sequence and head with masking
    logic_ptr = Logics + cur_head * stride_logic_h + (start_loc + col_offsets) * stride_logic_bs
    logits = tl.load(logic_ptr, mask=col_offsets < cur_seq_len, other=-float('inf')).to(tl.float32)
    
    # Compute numerically stable softmax
    max_logit = tl.max(logits, axis=0)
    logits_minus_max = logits - max_logit
    numerator = tl.exp(logits_minus_max)
    denominator = tl.sum(numerator, axis=0)
    probs = numerator / denominator
    
    # Store probabilities with masking
    prob_ptr = Prob_Out + cur_head * stride_prob_h + (start_loc + col_offsets) * stride_prob_bs
    tl.store(prob_ptr, probs, mask=col_offsets < cur_seq_len)

@torch.no_grad()
def token_softmax_fwd(Logics, B_Start_Loc, B_Seqlen, Prob_Out, max_input_len):
    # Determine optimal block size and execution parameters
    BLOCK_SIZE = triton.next_power_of_2(max_input_len)
    batch_size, num_heads = B_Start_Loc.shape[0], Logics.shape[0]
    
    # Configure number of warps based on block size
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 4096:
        num_warps = 16
    
    # Launch kernel with optimized parameters
    grid = (batch_size, num_heads)
    _fwd_kernel_token_softmax[grid](
        Logics, B_Start_Loc, B_Seqlen, Prob_Out,
        Logics.stride(0), Logics.stride(1),
        Prob_Out.stride(0), Prob_Out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
