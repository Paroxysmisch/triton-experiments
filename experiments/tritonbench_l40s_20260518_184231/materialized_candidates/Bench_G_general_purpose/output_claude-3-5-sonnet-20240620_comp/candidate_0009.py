import math
import torch
import triton
import triton.language as tl

# Constants
MAX_FUSED_SIZE = 65536
ROPE_GROUP_SIZE = 8

@triton.jit
def _rope_embedding(
    Q, Q_row_stride,
    cos, cos_row_stride,
    sin, sin_row_stride,
    seqlen, head_dim, n_heads,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROPE_GROUP_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Calculate row and group indices
    row_idx = pid // (n_heads // ROPE_GROUP_SIZE)
    group_idx = pid % (n_heads // ROPE_GROUP_SIZE)
    
    # Compute offsets
    q_offset = row_idx * Q_row_stride + group_idx * ROPE_GROUP_SIZE * head_dim
    cos_offset = row_idx * cos_row_stride
    sin_offset = row_idx * sin_row_stride
    
    # Create block pointers
    Q_ptr = Q + q_offset
    cos_ptr = cos + cos_offset
    sin_ptr = sin + sin_offset
    
    # Create mask for valid elements
    mask = tl.arange(0, BLOCK_SIZE) < head_dim
    
    # Load values
    q_real = tl.load(Q_ptr, mask=mask)
    q_imag = tl.load(Q_ptr + head_dim//2, mask=mask)
    cos_val = tl.load(cos_ptr, mask=mask)
    sin_val = tl.load(sin_ptr, mask=mask)
    
    # Compute rotation
    if BACKWARD_PASS:
        out_real = q_real * cos_val + q_imag * sin_val
        out_imag = q_imag * cos_val - q_real * sin_val
    else:
        out_real = q_real * cos_val - q_imag * sin_val
        out_imag = q_imag * cos_val + q_real * sin_val
    
    # Store results
    tl.store(Q_ptr, out_real, mask=mask)
    tl.store(Q_ptr + head_dim//2, out_imag, mask=mask)

def calculate_settings(n):
    """Calculate optimal block size and number of warps."""
    next_power_of_2 = 2 ** math.ceil(math.log2(n))
    if next_power_of_2 > MAX_FUSED_SIZE:
        raise RuntimeError(f"Input size {n} exceeds maximum allowed size {MAX_FUSED_SIZE}")
    
    num_warps = 4
    if next_power_of_2 >= 2048:
        num_warps = 8
    elif next_power_of_2 >= 4096:
        num_warps = 16
        
    return next_power_of_2, num_warps

def _rope_embedding_forward_impl(Q, cos, sin):
    """Forward pass implementation of RoPE."""
    batch_size, seqlen, n_heads, head_dim = Q.shape
    Q = Q.reshape(batch_size * seqlen, n_heads, head_dim)
    
    # Calculate settings
    BLOCK_SIZE, num_warps = calculate_settings(head_dim)
    n_groups = n_heads // ROPE_GROUP_SIZE
    
    # Launch kernel
    grid = (seqlen * n_groups,)
    _rope_embedding[grid](
        Q, Q.stride(0),
        cos, cos.stride(0),
        sin, sin.stride(0),
        seqlen, head_dim, n_heads,
        False, BLOCK_SIZE, ROPE_GROUP_SIZE,
        num_warps=num_warps
    )
    
    return Q.reshape(batch_size, seqlen, n_heads, head_dim)

def _rope_embedding_backward_impl(dY, cos, sin):
    """Backward pass implementation of RoPE."""
    batch_size, seqlen, n_heads, head_dim = dY.shape
    dY = dY.reshape(batch_size * seqlen, n_heads, head_dim)
    
    # Calculate settings
    BLOCK_SIZE, num_warps = calculate_settings(head_dim)
    n_groups = n_heads // ROPE_GROUP_SIZE
    
    # Launch kernel
    grid = (seqlen * n_groups,)
    _rope_embedding[grid](
        dY, dY.stride(0),
        cos, cos.stride(0),
        sin, sin.stride(0),
        seqlen, head_dim, n_heads,
        True, BLOCK_SIZE, ROPE_GROUP_SIZE,
        num_warps=num_warps
    )
    
    return dY.reshape(batch_size, seqlen, n_heads, head_dim)

class RoPE(torch.nn.Module):
    def forward(self, x, cos, sin):
        return _rope_embedding_forward_impl(x, cos, sin)
    
    def backward(self, grad_output, cos, sin):
        return _rope_embedding_backward_impl(grad_output, cos, sin)
