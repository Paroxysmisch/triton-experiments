import triton
import triton.language as tl
import torch

# Define group size for parallel processing of heads
ROPE_GROUP_SIZE = 4

@triton.jit
def _triton_rope(
    q_ptr, q_row_stride,
    k_ptr, k_row_stride,
    cos_ptr, cos_row_stride,
    sin_ptr, sin_row_stride,
    seqlen,
    head_dim: tl.constexpr,
    n_heads: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Applies Rotary Position Embedding to both query and key tensors.
    """
    # Each program instance processes a row and a group of heads
    row_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    
    # Offsets for the current block of columns
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim  # Mask for valid columns
    
    # Calculate position within the sequence
    pos = row_idx % seqlen
    # Load sine and cosine values for the current position
    cos = tl.load(cos_ptr + pos * cos_row_stride + col_offsets, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + pos * sin_row_stride + col_offsets, mask=mask, other=0.0)
    
    # Negate sine for backward pass to invert rotation
    if BACKWARD_PASS:
        sin = -sin
    
    # Determine the range of heads processed by this instance
    head_start = group_idx * ROPE_GROUP_SIZE
    head_end = tl.minimum(head_start + ROPE_GROUP_SIZE, n_heads)
    
    # Process each head in the current group
    for head in range(head_start, head_end):
        # Calculate offsets for Q and K
        q_offset = row_idx * q_row_stride + head * head_dim
        k_offset = row_idx * k_row_stride + head * head_dim
        
        # Offsets for the two halves of the head dimension
        off1 = q_offset + col_offsets
        off2 = q_offset + half_head_dim + col_offsets
        # Load Q values
        q1 = tl.load(q_ptr + off1, mask=mask, other=0.0).to(sin.dtype)
        q2 = tl.load(q_ptr + off2, mask=mask, other=0.0).to(sin.dtype)
        # Apply rotation and store Q
        tl.store(q_ptr + off1, q1 * cos - q2 * sin, mask=mask)
        tl.store(q_ptr + off2, q2 * cos + q1 * sin, mask=mask)
        
        # Repeat for K
        k_off1 = k_offset + col_offsets
        k_off2 = k_offset + half_head_dim + col_offsets
        k1 = tl.load(k_ptr + k_off1, mask=mask, other=0.0).to(sin.dtype)
        k2 = tl.load(k_ptr + k_off2, mask=mask, other=0.0).to(sin.dtype)
        tl.store(k_ptr + k_off1, k1 * cos - k2 * sin, mask=mask)
        tl.store(k_ptr + k_off2, k2 * cos + k1 * sin, mask=mask)

def rope_forward(q, k, cos, sin):
    # Ensure cos and sin are 2D (seq_len, head_dim)
    cos = cos.squeeze()
    sin = sin.squeeze()
    batch, seq_len, n_heads, head_dim = q.shape
    
    # Reshape Q and K to (batch*seq_len, n_heads*head_dim)
    q_flat = q.view(batch * seq_len, n_heads * head_dim)
    k_flat = k.view(batch * seq_len, n_heads * head_dim)
    n_rows, _ = q_flat.shape
    
    # Calculate block size and number of groups for kernel configuration
    BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)
    if BLOCK_SIZE < 16:
        BLOCK_SIZE = 16
    num_warps = 4 if BLOCK_SIZE >= 128 else 2
    
    # Determine the number of head groups
    n_groups = triton.cdiv(n_heads, ROPE_GROUP_SIZE)
    
    # Launch kernel for forward pass
    _triton_rope[(n_rows, n_groups)](
        q_flat, q_flat.stride(0),
        k_flat, k_flat.stride(0),
        cos, cos.stride(0),
        sin, sin.stride(0),
        seqlen=seq_len,
        head_dim=head_dim,
        n_heads=n_heads,
        BACKWARD_PASS=False,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    # Reshape back to original dimensions
    return (
        q_flat.view(batch, seq_len, n_heads, head_dim),
        k_flat.view(batch, seq_len, n_heads, head_dim),
    )

def rope_backward(dq, dk, cos, sin):
    # Transpose to match expected memory layout
    dq = dq.transpose(1, 2)
    dk = dk.transpose(1, 2)
    batch, seq_len, n_heads, head_dim = dq.shape
    
    # Flatten dimensions for kernel processing
    dq_flat = dq.reshape(batch * seq_len, n_heads * head_dim)
    dk_flat = dk.reshape(batch * seq_len, n_heads * head_dim)
    n_rows, _ = dq_flat.shape
    
    # Kernel configuration (same as forward)
    BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)
    if BLOCK_SIZE < 16:
        BLOCK_SIZE = 16
    num_warps = 4 if BLOCK_SIZE >= 128 else 2
    n_groups = triton.cdiv(n_heads, ROPE_GROUP_SIZE)
    
    # Launch kernel for backward pass
    _triton_rope[(n_rows, n_groups)](
        dq_flat, dq_flat.stride(0),
        dk_flat, dk_flat.stride(0),
        cos, cos.stride(0),
        sin, sin.stride(0),
        seqlen=seq_len,
        head_dim=head_dim,
        n_heads=n_heads,
        BACKWARD_PASS=True,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    # Transpose back and reshape
    dq = dq_flat.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    dk = dk_flat.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    return dq, dk

class RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        ctx.save_for_backward(cos, sin)
        q_out, k_out = rope_forward(q, k, cos, sin)
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        dq, dk = rope_backward(dq, dk, cos, sin)
        return dq, dk, None, None

def apply_rotary_emb(q, k, cos, sin):
    return RoPEFunction.apply(q, k, cos, sin)
