import triton
import triton.language as tl

@triton.jit
def _triton_rope(
    q_ptr, k_ptr, cos, sin, dq_ptr, dk_ptr, q_row, q_col, k_row, k_col, pid, BACKWARD_PASS,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, SEQ_LEN: tl.constexpr
):
    # Each thread is responsible for one element in the query and key matrices
    q_idx = tl.program_id(0)
    k_idx = tl.program_id(1)
    
    # Extract row and column indices
    q_row_idx = q_idx // q_col
    q_col_idx = q_idx % q_col
    k_row_idx = k_idx // k_col
    k_col_idx = k_idx % k_col
    
    # Compute the embedding split index
    split_idx = HEAD_DIM // 2
    
    # Compute the phase shift for the embedding
    cos_idx = q_col_idx // 2
    sin_idx = q_col_idx // 2
    
    # Load the cosine and sine values
    cos_val = cos[cos_idx]
    sin_val = sin[sin_idx]
    
    # Extract the relevant elements from the query and key matrices
    q1 = q_ptr[q_row_idx * q_col + q_col_idx]
    q2 = q_ptr[q_row_idx * q_col + q_col_idx + split_idx]
    k1 = k_ptr[k_row_idx * k_col + k_col_idx]
    k2 = k_ptr[k_row_idx * k_col + k_col_idx + split_idx]
    
    # Apply the rotation formula
    if BACKWARD_PASS:
        # Backward pass: apply inverse rotation
        q1_out = q1 * cos_val + q2 * sin_val
        q2_out = -q1 * sin_val + q2 * cos_val
        k1_out = k1 * cos_val + k2 * sin_val
        k2_out = -k1 * sin_val + k2 * cos_val
    else:
        # Forward pass: apply standard rotation
        q1_out = q1 * cos_val - q2 * sin_val
        q2_out = q1 * sin_val + q2 * cos_val
        k1_out = k1 * cos_val - k2 * sin_val
        k2_out = k1 * sin_val + k2 * cos_val
    
    # Store the results in the output matrices
    if q_col_idx < split_idx:
        q_ptr[q_row_idx * q_col + q_col_idx] = q1_out
        q_ptr[q_row_idx * q_col + q_col_idx + split_idx] = q2_out
        k_ptr[k_row_idx * k_col + k_col_idx] = k1_out
        k_ptr[k_row_idx * k_col + k_col_idx + split_idx] = k2_out
    
    # Handle gradients for backward pass
    if BACKWARD_PASS:
        # Transpose and apply inverse rotation for gradients
        dq1 = dq_ptr[q_row_idx * q_col + q_col_idx]
        dq2 = dq_ptr[q_row_idx * q_col + q_col_idx + split_idx]
        dk1 = dk_ptr[k_row_idx * k_col + k_col_idx]
        dk2 = dk_ptr[k_row_idx * k_col + k_col_idx + split_idx]
        
        dq1_out = dq1 * cos_val + dq2 * sin_val
        dq2_out = -dq1 * sin_val + dq2 * cos_val
        dk1_out = dk1 * cos_val + dk2 * sin_val
        dk2_out = -dk1 * sin_val + dk2 * cos_val
        
        dq_ptr[q_row_idx * q_col + q_col_idx] = dq1_out
        dq_ptr[q_row_idx * q_col + q_col_idx + split_idx] = dq2_out
        dk_ptr[k_row_idx * k_col + k_col_idx] = dk1_out
        dk_ptr[k_row_idx * k_col + k_col_idx + split_idx] = dk2_out
