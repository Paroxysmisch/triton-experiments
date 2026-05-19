import triton
import triton.language as tl

@triton.jit
def pairwise_distance_normalize_kernel(
    x1,
    x2,
    output,
    x1_row_stride,
    x2_row_stride,
    n_cols,
    dim,
    p_norm,
    eps_norm,
    eps_distance,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    col_id = tl.program_id(1)
    
    # Load elements from x1 and x2
    x1_elem = tl.load(x1 + row_id * x1_row_stride + col_id)
    x2_elem = tl.load(x2 + row_id * x2_row_stride + col_id)
    
    # Normalize elements
    x1_norm = x1_elem / (tl.math.pow(tl.math.abs(x1_elem), p_norm) + eps_norm)
    x2_norm = x2_elem / (tl.math.pow(tl.math.abs(x2_elem), p_norm) + eps_norm)
    
    # Calculate distance
    diff = x1_norm - x2_norm
    distance = tl.math.pow(tl.math.abs(diff), p_norm) + eps_distance
    
    # Store result
    tl.store(output + row_id * x1_row_stride + col_id, distance)
