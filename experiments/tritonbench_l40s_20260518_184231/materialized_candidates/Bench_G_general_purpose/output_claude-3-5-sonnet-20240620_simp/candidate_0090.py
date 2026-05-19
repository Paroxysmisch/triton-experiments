import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel(
    Logics, V, Out, B_Loc, B_Start_Loc, B_Seqlen,
    stride_b, stride_h, stride_s,
    B, H, N_CTX,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID gives us batch and head indices
    pid = tl.program_id(0)
    bid = pid // H
    hid = pid % H

    # Compute pointer offsets for current batch and head
    base_loc = B_Start_Loc[bid]
    seqlen = B_Seqlen[bid]
    
    # Initialize pointers
    offs_logic = bid * stride_b + hid * stride_h
    offs_v = bid * stride_b + hid * stride_h
    offs_out = bid * stride_b + hid * stride_h

    # Loop over sequence length in blocks
    for start_n in range(0, seqlen, BLOCK_SIZE):
        # Block size for current iteration
        block_size = min(BLOCK_SIZE, seqlen - start_n)
        
        # Load block of logits
        logic_block = tl.load(Logics + offs_logic + B_Loc[base_loc + start_n] * stride_s,
                            mask=start_n < seqlen, other=float("-inf"))
        
        # Compute softmax normalization
        max_logic = tl.max(logic_block, axis=0)
        logic_block = logic_block - max_logic
        exp_logic = tl.exp(logic_block)
        sum_exp = tl.sum(exp_logic, axis=0)
        softmax_logic = exp_logic / sum_exp

        # Load corresponding values and compute weighted sum
        for offs_n in range(0, block_size):
            v_idx = B_Loc[base_loc + start_n + offs_n]
            v = tl.load(V + offs_v + v_idx * stride_s)
            out = tl.sum(softmax_logic[offs_n] * v)
            
            # Store result
            tl.store(Out + offs_out + (start_n + offs_n) * stride_s, out)

# Wrapper function to launch kernel
def token_softmax_reducev_fwd(logics, v, b_loc, b_start_loc, b_seqlen):
    batch_size = b_seqlen.shape[0]
    n_heads = logics.shape[1]
    max_seqlen = logics.shape[-1]
    
    # Allocate output tensor
    out = torch.empty_like(v)
    
    # Configure kernel parameters
    BLOCK_SIZE = 128
    grid = (batch_size * n_heads,)
    
    # Launch kernel
    _fwd_kernel[grid](
        logics, v, out, b_loc, b_start_loc, b_seqlen,
        logics.stride(0), logics.stride(1), logics.stride(-1),
        batch_size, n_heads, max_seqlen,
        BLOCK_SIZE
    )
    
    return out
