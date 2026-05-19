import torch
import triton
import triton.language as tl

@triton.jit
def rotary_embedding_kernel(
    # Pointers to matrices
    q_ptr, k_ptr,           # Query and Key tensors [B, H, L, D]
    cos_ptr, sin_ptr,       # Cos/Sin position embeddings [L, D/2]
    output_q_ptr,           # Output query tensor
    output_k_ptr,           # Output key tensor
    # Matrix dimensions
    seq_len, head_dim,      # Sequence length and head dimension
    num_heads,              # Number of attention heads
    batch_size,             # Batch size
    BLOCK_SIZE: tl.constexpr,  # Number of elements to process per block
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate current batch, head, and sequence position
    batch_idx = pid // (num_heads * seq_len)
    head_idx = (pid % (num_heads * seq_len)) // seq_len
    seq_idx = pid % seq_len
    
    # Compute offsets
    q_offset = batch_idx * num_heads * seq_len * head_dim + \
               head_idx * seq_len * head_dim + \
               seq_idx * head_dim
    k_offset = q_offset  # Same layout for key tensor
    
    # Load chunks of head_dim elements
    for dim_idx in range(0, head_dim, BLOCK_SIZE):
        # Create block pointers
        q_block_ptr = q_ptr + q_offset + dim_idx
        k_block_ptr = k_ptr + k_offset + dim_idx
        
        # Load query and key values
        dim_mask = tl.arange(0, BLOCK_SIZE) < (head_dim - dim_idx)
        q = tl.load(q_block_ptr, mask=dim_mask)
        k = tl.load(k_block_ptr, mask=dim_mask)
        
        # Load rotation matrices (cos/sin)
        cos_block_ptr = cos_ptr + seq_idx * (head_dim // 2) + (dim_idx // 2)
        sin_block_ptr = sin_ptr + seq_idx * (head_dim // 2) + (dim_idx // 2)
        
        cos = tl.load(cos_block_ptr, mask=dim_mask[:BLOCK_SIZE//2])
        sin = tl.load(sin_block_ptr, mask=dim_mask[:BLOCK_SIZE//2])
        
        # Apply rotary embeddings
        # For even indices
        q_even = q[0::2]
        k_even = k[0::2]
        # For odd indices
        q_odd = q[1::2]
        k_odd = k[1::2]
        
        # Rotate vectors
        q_rot_even = q_even * cos - q_odd * sin
        q_rot_odd = q_odd * cos + q_even * sin
        k_rot_even = k_even * cos - k_odd * sin
        k_rot_odd = k_odd * cos + k_even * sin
        
        # Interleave results
        q_out = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        k_out = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        q_out[0::2] = q_rot_even
        q_out[1::2] = q_rot_odd
        k_out[0::2] = k_rot_even
        k_out[1::2] = k_rot_odd
        
        # Store results
        output_q_block_ptr = output_q_ptr + q_offset + dim_idx
        output_k_block_ptr = output_k_ptr + k_offset + dim_idx
        tl.store(output_q_block_ptr, q_out, mask=dim_mask)
        tl.store(output_k_block_ptr, k_out, mask=dim_mask)

# Python wrapper function
def apply_rotary_embedding(q, k, cos, sin):
    """
    Apply rotary embeddings to query and key tensors.
    
    Args:
        q: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
        cos: Cosine position embedding of shape [seq_len, head_dim//2]
        sin: Sine position embedding of shape [seq_len, head_dim//2]
    
    Returns:
        Tuple of rotary-embedded (query, key) tensors
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    
    # Allocate output tensors
    output_q = torch.empty_like(q)
    output_k = torch.empty_like(k)
    
    # Configure grid and block sizes
    BLOCK_SIZE = 32  # Can be tuned for performance
    grid = (batch_size * num_heads * seq_len,)
    
    # Launch kernel
    rotary_embedding_kernel[grid](
        q, k,
        cos, sin,
        output_q, output_k,
        seq_len, head_dim, num_heads, batch_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output_q, output_k
