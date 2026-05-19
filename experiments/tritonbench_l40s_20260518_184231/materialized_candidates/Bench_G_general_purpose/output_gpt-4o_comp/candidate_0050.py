import triton
import triton.language as tl
import torch

# Define the Triton kernel for ReLU
@triton.jit
def relu_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get the program ID, which determines the block of data this thread will process
    pid = tl.program_id(0)
    
    # Calculate the start of the block for this thread
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load the input data
    input_data = tl.load(input_ptr + offsets, mask=offsets < N, other=0.0)
    
    # Perform the ReLU operation
    relu_result = tl.where(input_data > 0, input_data, 0.0)
    
    # Store the result back to the output tensor
    tl.store(output_ptr + offsets, relu_result, mask=offsets < N)

# Define the wrapper function for the ReLU operation
def relu(input_tensor):
    # Get the number of elements in the input tensor
    N = input_tensor.numel()
    
    # Define the block size for processing
    BLOCK_SIZE = 1024
    
    # Allocate output tensor
    output_tensor = torch.empty_like(input_tensor)
    
    # Calculate the number of blocks needed
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Launch the Triton kernel
    relu_kernel[grid](input_tensor, output_tensor, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return output_tensor

# Example usage
if __name__ == "__main__":
    # Create a sample input tensor
    input_tensor = torch.tensor([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0], device='cuda')
    
    # Perform the ReLU operation using the Triton kernel
    output_tensor = relu(input_tensor)
    
    # Print the result
    print("Input Tensor:", input_tensor)
    print("ReLU Output Tensor:", output_tensor)
