import triton
import triton.language as tl

@triton.jit
def rotary_kernel(X_ptr, cos_ptr, sin_ptr, out_ptr, n_heads, head_dim, rotary_dim, seq_len, BLOCK_SIZE: tl.constexpr):
    # Obtain the block and thread indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)

    # Calculate the offsets
    offset = batch_idx * n_heads * seq_len * head_dim + head_idx * seq_len * head_dim + seq_idx * head_dim
    rotary_offset = offset + rotary_dim

    # Load the rotary dimension data
    x_rotary = tl.load(X_ptr + rotary_offset, mask=rotary_offset < seq_len * head_dim)
    cos_rotary = tl.load(cos_ptr + rotary_offset, mask=rotary_offset < seq_len * head_dim)
    sin_rotary = tl.load(sin_ptr + rotary_offset, mask=rotary_offset < seq_len * head_dim)

    # Apply rotary transformation
    x_rotated = x_rotary * cos_rotary + tl.math.fma(-x_rotary, sin_rotary, 0)

    # Store the result
    tl.store(out_ptr + rotary_offset, x_rotated, mask=rotary_offset < seq_len * head_dim)

### Python Wrapper: `apply_rotary`

This function will set up the input tensors, call the Triton kernel, and manage the execution parameters.
