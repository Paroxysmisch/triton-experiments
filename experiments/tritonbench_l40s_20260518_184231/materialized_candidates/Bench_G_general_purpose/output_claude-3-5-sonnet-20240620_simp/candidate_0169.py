import triton
import triton.language as tl
import torch

@triton.jit
def _copy_to_kcache_seqlen_n_kernel(
    # Pointers to input and output tensors
    K_ptr, KCache_ptr, block_tables_ptr,
    # Dimensions and strides
    batch_size, n_heads, head_dim, n_tokens,
    K_batch_stride, K_head_stride, K_token_stride,
    KCache_batch_stride, KCache_head_stride, KCache_block_stride, KCache_token_stride,
    block_tables_batch_stride,
    # Metadata
    blocks_per_seq: tl.constexpr,
    tokens_per_block: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    batch_id = pid // n_heads
    head_id = pid % n_heads

    # Compute base pointers
    K_base = K_ptr + batch_id * K_batch_stride + head_id * K_head_stride
    KCache_base = KCache_ptr + batch_id * KCache_batch_stride + head_id * KCache_head_stride
    block_table = block_tables_ptr + batch_id * block_tables_batch_stride

    # Handle each token in the sequence
    for token_idx in range(n_tokens):
        # Compute block and token position
        block_idx = token_idx // tokens_per_block
        token_in_block = token_idx % tokens_per_block
        
        # Load physical block index from block table
        physical_block = tl.load(block_table + block_idx)
        
        # Compute source and destination offsets
        src_offset = token_idx * K_token_stride
        dst_offset = (physical_block * KCache_block_stride + 
                     token_in_block * KCache_token_stride)

        # Load and store data in blocks of BLOCK_SIZE
        for idx in range(0, head_dim, BLOCK_SIZE):
            mask = idx + tl.arange(0, BLOCK_SIZE) < head_dim
            # Load from input
            x = tl.load(K_base + src_offset + idx, mask=mask)
            # Store to cache
            tl.store(KCache_base + dst_offset + idx, x, mask=mask)

def copy_k_to_blocked_cache(
    K: torch.Tensor,
    KCache: torch.Tensor,
    block_tables: torch.Tensor,
    blocks_per_seq: int,
    tokens_per_block: int
):
    """
    Copy keys/values to blocked cache format.
    
    Args:
        K: Input tensor of shape [batch_size, n_heads, n_tokens, head_dim]
        KCache: Output cache tensor
        block_tables: Block allocation table of shape [batch_size, blocks_per_seq]
        blocks_per_seq: Number of blocks per sequence
        tokens_per_block: Number of tokens per block
    """
    batch_size, n_heads, n_tokens, head_dim = K.shape
    
    # Compute strides
    K_batch_stride = K.stride(0)
    K_head_stride = K.stride(1)
    K_token_stride = K.stride(2)
    
    KCache_batch_stride = KCache.stride(0)
    KCache_head_stride = KCache.stride(1)
    KCache_block_stride = KCache.stride(2)
    KCache_token_stride = KCache.stride(3)
    
    block_tables_batch_stride = block_tables.stride(0)

    # Configure block size based on head dimension
    BLOCK_SIZE = min(128, triton.next_power_of_2(head_dim))
    
    # Launch kernel
    grid = (batch_size * n_heads,)
    _copy_to_kcache_seqlen_n_kernel[grid](
        K.data_ptr(), KCache.data_ptr(), block_tables.data_ptr(),
        batch_size, n_heads, head_dim, n_tokens,
        K_batch_stride, K_head_stride, K_token_stride,
        KCache_batch_stride, KCache_head_stride, KCache_block_stride, KCache_token_stride,
        block_tables_batch_stride,
        blocks_per_seq=blocks_per_seq,
        tokens_per_block=tokens_per_block,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4 if head_dim > 512 else 2,
    )
