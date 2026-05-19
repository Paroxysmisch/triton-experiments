import torch
import triton
import triton.language as tl

@triton.jit
def _copy_to_kcache_seqlen_n_kernel(
    K,                      # Input tensor (K or V)
    KCache,                 # Cache tensor
    BLOCK_TABLES,          # Block mapping table
    seq_lengths,           # Sequence lengths
    stride_kt,             # Stride for token dimension in K
    stride_kh,             # Stride for head dimension in K
    stride_kd,             # Stride for hidden dimension in K
    stride_kcb,            # Stride for block dimension in cache
    stride_kch,            # Stride for head dimension in cache
    stride_kcsplit_x,      # Stride for split dimension in cache
    stride_kcs,            # Stride for sequence dimension in cache
    stride_kcx,            # Stride for hidden dimension in cache
    stride_bts,            # Stride for sequence dimension in block table
    stride_btb,            # Stride for block dimension in block table
    block_size,            # Size of each block
    n_tokens,              # Number of tokens per sequence
    HEAD_DIM: tl.constexpr,
    KCACHE_X: tl.constexpr,
):
    # Get program IDs for parallel execution
    pid_token = tl.program_id(0)  # Token index
    pid_head = tl.program_id(1)   # Head index
    pid_split = tl.program_id(2)  # Split index for hidden dimension
    
    # Calculate sequence index and token shift
    seq_idx = pid_token // n_tokens
    token_shift = pid_token % n_tokens
    
    # Load sequence length and calculate block position
    seq_len = tl.load(seq_lengths + seq_idx) + token_shift
    block_idx = seq_len // block_size
    
    # Get block ID from block table
    block_ptr = BLOCK_TABLES + seq_idx * stride_bts
    block_id = tl.load(block_ptr + block_idx * stride_btb)
    
    # Calculate offset within block
    block_offset = seq_len % block_size
    
    # Create offset arrays for loading and storing
    offs_dim = pid_split * KCACHE_X + tl.arange(0, KCACHE_X)
    
    # Calculate input offsets
    offs_k = (pid_token * stride_kt + 
             pid_head * stride_kh + 
             offs_dim * stride_kd)
    
    # Load input data
    k_data = tl.load(K + offs_k)
    
    # Calculate cache offsets
    offs_cache = (block_id * stride_kcb +
                 pid_head * stride_kch +
                 pid_split * stride_kcsplit_x +
                 block_offset * stride_kcs +
                 tl.arange(0, KCACHE_X))
    
    # Store to cache
    tl.store(KCache + offs_cache, k_data)
