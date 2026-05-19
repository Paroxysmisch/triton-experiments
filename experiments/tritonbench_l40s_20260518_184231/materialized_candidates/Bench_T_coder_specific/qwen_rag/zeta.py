import triton
import triton.language as tl

@triton.jit
def hurwitz_zeta_kernel(
    x_ptr,
    q_ptr,
    output_ptr,
    N,
    M,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr
):
    # Get the row and column indices for the current program
    row_id = tl.program_id(axis=0)
    col_id = tl.program_id(axis=1)
    
    # Calculate the starting point for k in the series
    k_start = row_id * BLOCK_SIZE_X + col_id
    
    # Initialize the sum for the current element
    partial_sum = tl.zeros([], dtype=tl.float32)
    
    # Loop over the series terms up to a large enough value to converge
    for k in range(BLOCK_SIZE_X):
        term = tl.pow(tl.f32(k + q_ptr[row_id]), -x_ptr[row_id])
        partial_sum += term
    
    # Store the partial sum in the output pointer
    tl.atomic_add(output_ptr[row_id], partial_sum)
