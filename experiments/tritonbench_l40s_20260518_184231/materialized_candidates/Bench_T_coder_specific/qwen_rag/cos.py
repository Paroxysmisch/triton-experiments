import triton
import triton.language as tl
import torch

# Define the Triton kernel to compute the cosine of elements in a tensor
@triton.jit
def cos_kernel(
    X_ptr,  # Pointer to the input tensor
    Y_ptr,  # Pointer to the output tensor
    N,      # Number of elements in the input tensor
    BLOCK_SIZE: tl.constexpr  # Block size for parallel execution
):
    # Determine the global ID within the block
    x_idx = tl.program_id(axis=0)
    # Calculate the starting offset for this block
    start = x_idx * BLOCK_SIZE
    # Determine how many elements this block will process
    count = min(BLOCK_SIZE, N - start)

    # Load data into shared memory
    x_shared = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    x_shared[:count] = tl.load(X_ptr + start, mask=count > 0)

    # Synchronize all threads in the block to ensure all loads complete
    tl.barrier()

    # Compute the cosine for each element in the block
    y = tl.cos(x_shared)

    # Store the results back to global memory
    tl.store(Y_ptr + start, y, mask=count > 0)

# Define the wrapper function to use the Triton kernel
def cos(input_tensor, out=None):
    # Get the number of elements in the input tensor
    N = input_tensor.size()
    
    # Check if an output tensor was provided
    if out is None:
        out = torch.empty_like(input_tensor)
    
    # Ensure both input and output tensors are on the same device
    assert input_tensor.device == out.device, 'Input and output tensors must be on the same device'
    
    # Choose a block size for parallel execution
    BLOCK_SIZE = triton.next_power_of_2(N // 4)
    
    # Launch the Triton kernel
    cos_kernel[(N + BLOCK_SIZE - 1) // BLOCK_SIZE, BLOCK_SIZE](
        X_ptr=input_tensor.data_ptr(),
        Y_ptr=out.data_ptr(),
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Example usage
if __name__ == "__main__":
    input_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0], device='cuda')
    output_tensor = cos(input_tensor)
    print(output_tensor)
