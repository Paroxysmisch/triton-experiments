triton
import triton
import triton.language as tl

@triton.jit
def _triton_rope(q_ptr, k_ptr, cos_ptr, sin_ptr, q_row_stride, k_row_stride, cos_row_stride, sin_row_stride, batch_size, seq_len, head_dim, BACKWARD_PASS):
    # Get the program ID (pid) which uniquely identifies each thread
    pid = tl.program_id(axis=0)
    
    # Calculate the row and column indices
    row = pid // seq_len
    col = pid % seq_len
    
    # Calculate the global indices for q and k matrices
    q_global_idx = row * q_row_stride + col
    k_global_idx = row * k_row_stride + col
    
    # Load q and k slices
    q_slice = tl.load(q_ptr + q_global_idx, mask=col < seq_len, other=0.0)
    k_slice = tl.load(k_ptr + k_global_idx, mask=col < seq_len, other=0.0)
    
    # Load cosine and sine values
    cos_idx = col % head_dim
    sin_idx = (col + head_dim // 2) % head_dim
    cos_val = tl.load(cos_ptr + cos_idx * cos_row_stride, mask=col < seq_len, other=0.0)
    sin_val = tl.load(sin_ptr + sin_idx * sin_row_stride, mask=col < seq_len, other=0.0)
    
    # Apply rotary transformation
    if BACKWARD_PASS:
        # Backward pass: apply inverse rotation
        q_rotated = q_slice * cos_val + k_slice * sin_val
        k_rotated = -q_slice * sin_val + k_slice * cos_val
    else:
        # Forward pass: apply rotation
        q_rotated = q_slice * cos_val - k_slice * sin_val
        k_rotated = q_slice * sin_val + k_slice * cos_val
    
    # Store the transformed slices back
    tl.store(q_ptr + q_global_idx, q_rotated, mask=col < seq_len)
    tl.store(k_ptr + k_global_idx, k_rotated, mask=col < seq_len)
