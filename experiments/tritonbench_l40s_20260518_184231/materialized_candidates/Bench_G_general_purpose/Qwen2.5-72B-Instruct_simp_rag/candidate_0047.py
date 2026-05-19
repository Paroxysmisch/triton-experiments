import triton
import triton.language as tl
import torch

# Define the ReLU kernel
@triton.jit
def relu_kernel(x_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    # Compute the block index
    pid = tl.program_id(axis=0)
    
    # Compute the starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create a range of indices for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to filter out-of-bounds indices
    mask = offsets < numel
    
    # Load the input data
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Apply the ReLU operation
    x = tl.where(x > 0, x, 0)
    
    # Store the result back to the output tensor
    tl.store(out_ptr + offsets, x, mask=mask)

# Define the wrapper function
def run_triton(x, out):
    # Ensure the input and output tensors are on the GPU
    x = x.to(device="cuda")
    out = out.to(device="cuda")
    
    # Get the number of elements in the tensor
    numel = x.numel()
    
    # Set the grid and block dimensions
    grid = (numel + 1023) // 1024  # Ensure all elements are covered
    
    # Launch the kernel
    relu_kernel[grid, 1024](x, out, numel, BLOCK_SIZE=1024)
    
    return out

# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    x = torch.randn((67108864, 4), device="cuda")
    
    # Create an output tensor of the same shape
    out = torch.empty_like(x, device="cuda")
    
    # Run the ReLU operation
    out = run_triton(x, out)
    
    # Print the first few elements of the output tensor
    print(out[:10])
