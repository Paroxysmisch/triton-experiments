import triton
import triton.language as tl
import torch

@triton.jit
def relu_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Block size (compile-time constant)
):
    # Compute the program ID
    pid = tl.program_id(0)
    
    # Compute the start index for this block
    block_start = pid * BLOCK_SIZE
    
    # Compute the offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid elements
    mask = offsets < n_elements
    
    # Load input elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Apply ReLU operation
    output = tl.maximum(x, 0.0)
    
    # Store the result
    tl.store(out_ptr + offsets, output, mask=mask)

def relu(x):
    # Get the shape of the input tensor
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # Define the grid
    grid = (triton.cdiv(n_elements, 1024),)
    
    # Launch the kernel
    relu_kernel[grid](
        x, output,
        n_elements,
        BLOCK_SIZE=1024,
    )
    
    return output

# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    x = torch.randn(1000000, device='cuda', dtype=torch.float32)
    
    # Apply ReLU using our Triton kernel
    result = relu(x)
    
    # Verify the result
    torch_result = torch.nn.functional.relu(x)
    assert torch.allclose(result, torch_result), "Results do not match!"
    print("ReLU operation successful and verified!")
