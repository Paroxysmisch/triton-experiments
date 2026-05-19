triton
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_online_v2(
    X_ptr: tl.tensor, 
    Y_ptr: tl.tensor, 
    M: tl.int32, 
    N: tl.int32, 
    TILE_N: tl.int32
):
    """
    Triton-optimized kernel to compute the softmax function over a 2D input tensor.
    
    Parameters:
    - X_ptr: Pointer to the input data.
    - Y_ptr: Pointer to the output data.
    - M: Number of rows in the input matrix.
    - N: Number of columns in the input matrix.
    - TILE_N: Tile size for tiling the computation.
    """
    # Compute the row and column indices
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Compute the tile-aligned boundary
    row_start = tl.prev_multiple_of(row * TILE_N, TILE_N)
    row_end = min(row_start + TILE_N, M)
    
    # Reduction phase to compute max value and sum of exponentials
    max_val = -tl.inf
    sum_exp = 0.0
    for i in range(row_start, row_end):
        max_val = tl.maximum(max_val, X_ptr[i * N + col])
        exp_val = tl.exp(X_ptr[i * N + col] - max_val)
        sum_exp += exp_val
    
    # Final output phase
    for i in range(row_start, row_end):
        exp_val = tl.exp(X_ptr[i * N + col] - max_val)
        Y_ptr[i * N + col] = exp_val / sum_exp
