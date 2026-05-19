import torch
import triton
import triton.language as tl

@triton.jit
def _triton_rope(
    # Pointers to matrices
    q_ptr, k_ptr, cos_ptr, sin_ptr, 
    output_q_ptr, output_k_ptr,
    # Matrix dimensions
    batch_size, seq_len, num_heads, head_dim,
    # Strides for the different dimensions
    stride_b, stride_s, stride_h, stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    
    # Calculate indices for batch, sequence, and head
    batch_idx = pid // (seq_len * num_heads)
    tmp = pid % (seq_len * num_heads)
    seq_idx = tmp // num_heads
    head_idx = tmp % num_heads

    # Compute base offsets for the current block
    base_q_offset = (
        batch_idx * stride_b + 
        seq_idx * stride_s + 
        head_idx * stride_d
    )
    
    # Load a block of elements
    offs_d = tl.arange(0, BLOCK_SIZE)
    mask = offs_d < head_dim
    
    # Load q and k vectors for current position
    q = tl.load(q_ptr + base_q_offset + offs_d, mask=mask)
    k = tl.load(k_ptr + base_q_offset + offs_d, mask=mask)
    
    # Load rotation matrices
    cos = tl.load(cos_ptr + seq_idx * head_dim + offs_d, mask=mask)
    sin = tl.load(sin_ptr + seq_idx * head_dim + offs_d, mask=mask)
    
    # Apply rotation - note we handle pairs of elements
    q_rot_even = q * cos - tl.roll(q, 1) * sin
    q_rot_odd = q * sin + tl.roll(q, -1) * cos
    k_rot_even = k * cos - tl.roll(k, 1) * sin
    k_rot_odd = k * sin + tl.roll(k, -1) * cos
    
    # Combine even and odd elements
    q_out = tl.where(offs_d % 2 == 0, q_rot_even, q_rot_odd)
    k_out = tl.where(offs_d % 2 == 0, k_rot_even, k_rot_odd)
    
    # Store results
    tl.store(output_q_ptr + base_q_offset + offs_d, q_out, mask=mask)
    tl.store(output_k_ptr + base_q_offset + offs_d, k_out, mask=mask)

def rope_forward(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Apply rotary position embedding to query and key tensors.
    
    Args:
        q: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
        k: Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
        cos: Cosine tensor of shape (seq_len, head_dim)
        sin: Sine tensor of shape (seq_len, head_dim)
    
    Returns:
        Tuple of transformed (q, k) tensors
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Ensure all inputs are contiguous and in the correct dtype
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    
    # Create output tensors
    output_q = torch.empty_like(q)
    output_k = torch.empty_like(k)
    
    # Calculate strides
    stride_b = q.stride(0)
    stride_s = q.stride(1)
    stride_h = q.stride(2)
    stride_d = q.stride(3)
    
    # Configure block size - should be multiple of 8 for good performance
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    
    # Launch kernel
    grid = (batch_size * seq_len * num_heads,)
    _triton_rope[grid](
        q, k, cos, sin,
        output_q, output_k,
        batch_size, seq_len, num_heads, head_dim,
        stride_b, stride_s, stride_h, stride_d,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_q, output_k
