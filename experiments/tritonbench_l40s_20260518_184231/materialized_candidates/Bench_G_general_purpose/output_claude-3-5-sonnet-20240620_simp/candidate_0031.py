import triton
import triton.language as tl
import torch

# Define the Triton kernel
@triton.jit
def mul_kernel(
    dst_ptr,
    src_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(axis=0)
    
    # Compute the start index for this block
    block_start = pid * BLOCK_SIZE
    
    # Compute the offset for each thread within the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where n_elements is not a multiple of BLOCK_SIZE
    mask = offsets < n_elements
    
    # Load data from src
    x = tl.load(src_ptr + offsets, mask=mask)
    
    # Define the constant exponent compensator
    compensator = tl.full(x.dtype, 2.0 ** (127 - 15))
    
    # Perform the multiplication
    output = x * compensator
    
    # Store the result in dst
    tl.store(dst_ptr + offsets, output, mask=mask)

# Define the wrapper function
def mul_wrapper(src: torch.Tensor) -> torch.Tensor:
    # Ensure input is on GPU
    assert src.is_cuda, "Input tensor must be on GPU"
    
    # Create output tensor with same shape and dtype as input
    dst = torch.empty_like(src)
    
    # Define the block size (can be tuned for performance)
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(src.numel(), BLOCK_SIZE),)
    
    # Launch the kernel
    mul_kernel[grid](
        dst,
        src,
        src.numel(),
        BLOCK_SIZE,
    )
    
    return dst

# Example usage
if __name__ == "__main__":
    # Create a sample input tensor
    x = torch.randn(1000000, device='cuda')
    
    # Call the wrapper function
    result = mul_wrapper(x)
    
    # Verify the result
    expected = x * (2.0 ** (127 - 15))
    assert torch.allclose(result, expected), "Results do not match!"
    print("Verification passed!")
