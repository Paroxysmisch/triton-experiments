import triton
import triton.language as tl
import torch

# Define the Triton kernel for the ReLU operation
@triton.jit
def relu_kernel(x_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    # Compute the index for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go out of bounds
    mask = offsets < numel
    
    # Load data from the input pointer
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Apply the ReLU operation
    relu_x = tl.where(x > 0, x, 0.0)
    
    # Store the result to the output pointer
    tl.store(out_ptr + offsets, relu_x, mask=mask)

# Wrapper function to execute the Triton kernel
def run_relu(x):
    # Ensure the input tensor is on the GPU
    x = x.to(device='cuda')
    
    # Initialize an output tensor on the GPU
    y = torch.empty_like(x, device='cuda')
    
    # Define the block size for the kernel
    BLOCK_SIZE = 1024
    
    # Calculate the number of elements in the tensor
    numel = x.numel()
    
    # Launch the Triton kernel
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    relu_kernel[grid](x, y, numel, BLOCK_SIZE=BLOCK_SIZE)
    
    return y

# Example usage
x = torch.randn((67108864,), device='cuda')
y = run_relu(x)
