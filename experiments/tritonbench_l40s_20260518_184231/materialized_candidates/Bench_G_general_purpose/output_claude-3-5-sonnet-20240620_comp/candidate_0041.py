import triton
import triton.language as tl
import torch

@triton.jit
def _triton_rope(
    q_ptr, k_ptr,  # pointers to matrices
    cos_ptr, sin_ptr,  # pointers to rotation vectors
    q_stride_row, k_stride_row,  # row strides
    cos_stride_row, sin_stride_row,  # rotation strides
    batch_size, seq_len, head_dim,  # dimensions
    BLOCK_SIZE: tl.constexpr,  # block size for parallelization
    BACKWARD_PASS: tl.constexpr,  # flag for forward/backward pass
):
    # Get program ID and compute row index
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len

    # Compute starting offsets
    q_start = batch_idx * q_stride_row + seq_idx * head_dim
    k_start = batch_idx * k_stride_row + seq_idx * head_dim
    rot_start = seq_idx * head_dim

    # Create offsets for the current block
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < head_dim

    # Load data for the current block
    q = tl.load(q_ptr + q_start + offs, mask=mask)
    k = tl.load(k_ptr + k_start + offs, mask=mask)
    cos = tl.load(cos_ptr + rot_start + offs, mask=mask)
    sin = tl.load(sin_ptr + rot_start + offs, mask=mask)

    # Apply rotary embeddings
    # For even indices
    q_even = q[::2]
    q_odd = q[1::2]
    k_even = k[::2]
    k_odd = k[1::2]
    cos_even = cos[::2]
    sin_even = sin[::2]

    if not BACKWARD_PASS:
        # Forward pass rotation
        q_out_even = q_even * cos_even - q_odd * sin_even
        q_out_odd = q_odd * cos_even + q_even * sin_even
        k_out_even = k_even * cos_even - k_odd * sin_even
        k_out_odd = k_odd * cos_even + k_even * sin_even
    else:
        # Backward pass rotation (inverse)
        q_out_even = q_even * cos_even + q_odd * sin_even
        q_out_odd = q_odd * cos_even - q_even * sin_even
        k_out_even = k_even * cos_even + k_odd * sin_even
        k_out_odd = k_odd * cos_even - k_even * sin_even

    # Interleave results back
    q_out = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    k_out = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    q_out[::2] = q_out_even
    q_out[1::2] = q_out_odd
    k_out[::2] = k_out_even
    k_out[1::2] = k_out_odd

    # Store results
    tl.store(q_ptr + q_start + offs, q_out, mask=mask)
    tl.store(k_ptr + k_start + offs, k_out, mask=mask)

def rope_forward(q, k, cos, sin, backward=False):
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
        cos: Cosine rotation tensor of shape [seq_len, head_dim]
        sin: Sine rotation tensor of shape [seq_len, head_dim]
        backward: Whether to apply inverse rotation for backward pass
    
    Returns:
        Transformed q and k tensors
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Ensure inputs are contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    # Compute grid and block sizes
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    grid = (batch_size * seq_len,)

    # Launch kernel
    _triton_rope[(grid)](
        q_ptr=q.data_ptr(),
        k_ptr=k.data_ptr(),
        cos_ptr=cos.data_ptr(),
        sin_ptr=sin.data_ptr(),
        q_stride_row=q.stride(0),
        k_stride_row=k.stride(0),
        cos_stride_row=cos.stride(0),
        sin_stride_row=sin.stride(0),
        batch_size=batch_size,
        seq_len=seq_len,
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        BACKWARD_PASS=backward,
    )

    return q, k, cos, sin
