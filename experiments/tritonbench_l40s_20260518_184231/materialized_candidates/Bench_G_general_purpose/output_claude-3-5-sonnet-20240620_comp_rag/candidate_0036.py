import torch
import triton
import triton.language as tl

@triton.jit
def _triton_rope(
    # Pointers to matrices
    q_ptr, k_ptr,
    cos_ptr, sin_ptr,
    # Dimensions
    batch_size, seq_len, n_heads, head_dim,
    # Strides
    stride_q_b, stride_q_s, stride_q_h, stride_q_d,
    stride_k_b, stride_k_s, stride_k_h, stride_k_d,
    # Constants
    BLOCK_SIZE: tl.constexpr,
    BACKWARD_PASS: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate batch and sequence indices
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Early exit if out of bounds
    if batch_idx >= batch_size:
        return
        
    # Calculate offsets
    q_offset = (batch_idx * stride_q_b + seq_idx * stride_q_s)
    k_offset = (batch_idx * stride_k_b + seq_idx * stride_k_s)
    
    # Load block
    dims = tl.arange(0, BLOCK_SIZE)
    
    # Load q and k slices
    q = tl.load(q_ptr + q_offset + dims * stride_q_d)
    k = tl.load(k_ptr + k_offset + dims * stride_k_d)
    
    # Load rotation matrices
    cos = tl.load(cos_ptr + seq_idx * BLOCK_SIZE + dims)
    sin = tl.load(sin_ptr + seq_idx * BLOCK_SIZE + dims)
    
    # Apply rotation
    if not BACKWARD_PASS:
        # Forward pass rotation
        q_rot = q * cos - tl.roll(q, 1) * sin
        k_rot = k * cos - tl.roll(k, 1) * sin
    else:
        # Backward pass rotation (conjugate)
        q_rot = q * cos + tl.roll(q, 1) * sin
        k_rot = k * cos + tl.roll(k, 1) * sin
    
    # Store results
    tl.store(q_ptr + q_offset + dims * stride_q_d, q_rot)
    tl.store(k_ptr + k_offset + dims * stride_k_d, k_rot)

def rope_forward(q: torch.Tensor, k: torch.Tensor, 
                cos: torch.Tensor, sin: torch.Tensor,
                backward: bool = False):
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor of shape [batch_size, seq_len, n_heads, head_dim]
        k: Key tensor of shape [batch_size, seq_len, n_heads, head_dim]
        cos: Cosine rotation matrix of shape [seq_len, head_dim]
        sin: Sine rotation matrix of shape [seq_len, head_dim]
        backward: Whether this is a backward pass
    """
    
    # Get dimensions
    batch_size, seq_len, n_heads, head_dim = q.shape
    
    # Ensure inputs are contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    
    # Calculate grid and block sizes
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    grid = (batch_size * seq_len,)
    
    # Launch kernel
    _triton_rope[(grid,)](
        q, k,
        cos, sin,
        batch_size, seq_len, n_heads, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
        BACKWARD_PASS=backward
    )
    
    return q, k, cos, sin
