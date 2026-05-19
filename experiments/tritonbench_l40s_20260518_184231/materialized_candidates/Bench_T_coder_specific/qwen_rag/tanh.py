import triton
import triton.language as tl

@triton.jit
def tanh_kernel(
    X_ptr,  # Pointer to the input tensor
    Y_ptr,  # Pointer to the output tensor
    N,      # Number of elements in the input tensor
    BLOCK_SIZE: tl.constexpr  # Block size for parallel execution
):
    # Get the global index for the current thread
    x_idx = tl.program_id(axis=0)
    
    # Calculate the starting index for this block
    start_idx = x_idx * BLOCK_SIZE
    
    # Load data into shared memory
    x_shared = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    mask = start_idx < N
    x_shared[0:BLOCK_SIZE] = tl.load(X_ptr + start_idx, mask=mask)
    
    # Perform tanh computation in shared memory
    y_shared = tl.tanh(x_shared)
    
    # Write the result back to global memory
    tl.store(Y_ptr + start_idx, y_shared, mask=mask)
