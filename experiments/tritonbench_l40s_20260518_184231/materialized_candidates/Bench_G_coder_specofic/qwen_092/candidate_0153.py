triton
import triton
import triton.language as tl

# Define the kernel function
@triton.jit
def sin_kernel(
    in_ptr0: tl.tensor,  # Pointer to the input data
    out_ptr: tl.tensor, # Pointer to the output data
    n_elements: tl.int32, # Total number of elements to process
    BLOCK_SIZE: tl.constexpr, # Block size per program instance
):
    # Get the program ID along the first axis
    pid = tl.program_id(axis=0)
    
    # Calculate the starting index for this block
    start_idx = pid * BLOCK_SIZE
    
    # Calculate the ending index for this block
    end_idx = start_idx + BLOCK_SIZE
    
    # Ensure we do not exceed the total number of elements
    end_idx = min(end_idx, n_elements)
    
    # Compute the range of indices to process in this block
    indices = tl.arange(start_idx, end_idx)
    
    # Load input data with masking to prevent out-of-bounds access
    x = tl.load(in_ptr0 + indices, mask=indices < n_elements)
    
    # Compute the sine of each element
    y = tl.sin(x)
    
    # Store the result back into memory with masking
    tl.store(out_ptr + indices, y, mask=indices < n_elements)
