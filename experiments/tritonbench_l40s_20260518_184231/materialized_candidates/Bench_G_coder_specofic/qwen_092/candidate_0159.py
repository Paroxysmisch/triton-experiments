import triton
import triton.language as tl

@triton.jit
def nested3(
    in_ptr: tl.tensor,  # Input tensor
    out_ptr: tl.tensor, # Output tensor
    stride_m: tl.int32, # Stride for rows
    stride_n: tl.int32, # Stride for columns
    n_rows: tl.int32,   # Number of rows
    n_cols: tl.int32,   # Number of columns
    BLOCK_SIZE: tl.constexpr = 32  # Block size
):
    # Define block and grid indices
    pid = tl.program_id(axis=0)
    num_blocks = tl.cdiv(n_cols, BLOCK_SIZE)
    
    # Calculate the row and column indices for the current block
    row = pid // num_blocks
    col = pid % num_blocks
    
    # Define indices for the nested loops
    i = row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j = col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    k = tl.arange(0, BLOCK_SIZE)
    
    # Initialize pointers for input and output
    a_ptrs = in_ptr + (i[:, None] * stride_m + j[None, :] * stride_n + k[:, None])
    c_ptrs = out_ptr + (i[:, None] * stride_m + j[None, :] * stride_n + k[:, None])
    
    # Load values from input pointers
    a_vals = tl.load(a_ptrs)
    
    # Store values to output pointers
    tl.store(c_ptrs, a_vals)
