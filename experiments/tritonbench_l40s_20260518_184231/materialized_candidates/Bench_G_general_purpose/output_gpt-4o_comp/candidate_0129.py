import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_apply_penalty(
    Logits, 
    Penalties, 
    TokenIds, 
    TokenCounts, 
    CumsumSeqLen, 
    cur_batch, 
    stride_logits, 
    BLOCK: tl.constexpr
):
    # Compute the starting index for the current batch
    start_idx = tl.load(CumsumSeqLen + cur_batch)
    end_idx = tl.load(CumsumSeqLen + cur_batch + 1)
    
    # Compute the range of indices for the current batch
    range_size = end_idx - start_idx
    
    # Load penalty coefficients for the current batch
    presence_penalty = tl.load(Penalties + cur_batch * 3 + 0)
    frequency_penalty = tl.load(Penalties + cur_batch * 3 + 1)
    repetition_penalty = tl.load(Penalties + cur_batch * 3 + 2)
    
    # Load token IDs and their counts
    for idx in range(0, range_size, BLOCK):
        offset = start_idx + idx + tl.arange(0, BLOCK)
        
        # Load token ids and counts
        token_ids = tl.load(TokenIds + offset, mask=offset < end_idx, other=0)
        token_counts = tl.load(TokenCounts + offset, mask=offset < end_idx, other=0)
        
        # Adjust logits based on penalties
        logits_idx = cur_batch * stride_logits + token_ids
        logits = tl.load(Logits + logits_idx, mask=offset < end_idx, other=0.0)
        
        # Apply repetition penalty
        logits -= repetition_penalty * token_counts
        
        # Apply frequency penalty
        logits -= frequency_penalty * tl.log(token_counts + 1)
        
        # Apply presence penalty
        logits -= presence_penalty * (token_counts > 0)
        
        # Store adjusted logits back
        tl.store(Logits + logits_idx, logits, mask=offset < end_idx)


import torch

def next_power_of_2(x):
    return 1 << (x - 1).bit_length()

def apply_penalty(Logits, Penalties, TokenIds, TokenCounts, CumsumSeqLen):
    assert Logits.is_contiguous(), "Logits must be contiguous"
    
    # Determine the block size
    BLOCK = max(32, next_power_of_2(Logits.shape[1]))
    
    # Get the stride of the logits tensor
    stride_logits = Logits.stride(0)
    
    # Launch the kernel for each batch
    for cur_batch in range(Logits.shape[0]):
        _fwd_kernel_apply_penalty[grid=(1,)](
            Logits,
            Penalties,
            TokenIds,
            TokenCounts,
            CumsumSeqLen,
            cur_batch,
            stride_logits,
            BLOCK=BLOCK,
            num_warps=8
        )
