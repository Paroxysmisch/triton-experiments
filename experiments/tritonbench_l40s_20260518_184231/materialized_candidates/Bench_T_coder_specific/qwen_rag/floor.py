import triton
import triton.language as tl

@triton.jit
def floor_kernel(
    X_ptr, 
    Y_ptr, 
    N, 
    BLOCK_SIZE: tl.constexpr
):
    # Determine the index of the current thread within the block
    x_idx = tl.program_id(axis=0)
    # Determine the starting index of the current block
    start_idx = x_idx * BLOCK_SIZE
    # Generate indices for the current block
    idx = start_idx + tl.arange(0, BLOCK_SIZE)

    # Mask to ensure we don't access out-of-bounds memory
    mask = idx < N

    # Load values from the input tensor
    x = tl.load(X_ptr + idx, mask=mask)
    
    # Compute the floor of each element
    y = tl.floor(x)
    
    # Store the results in the output tensor
    tl.store(Y_ptr + idx, y, mask=mask)
