import torch
import triton
import triton.language as tl

@triton.jit
def mul_kernel(
    src_ptr,  # Pointer to source tensor
    dst_ptr,  # Pointer to destination tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Size of parallel execution block
):
    # Calculate the absolute position in the tensor
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for bounds checking
    mask = offsets < n_elements
    
    # Define the exponent compensator constant
    COMPENSATOR = tl.constexpr(2.0 ** (127 - 15))
    
    # Load elements from source tensor with bounds checking
    x = tl.load(src_ptr + offsets, mask=mask)
    
    # Multiply by compensator
    output = x * COMPENSATOR
    
    # Store results back to destination tensor
    tl.store(dst_ptr + offsets, output, mask=mask)

def launch_mul_kernel(src: torch.Tensor, BLOCK_SIZE: int = 1024):
    """
    Wrapper function to launch the multiplication kernel
    
    Args:
        src: Input tensor on CUDA device
        BLOCK_SIZE: Block size for parallel execution
    
    Returns:
        dst: Output tensor containing results
    """
    # Input validation
    assert src.is_cuda, "Input tensor must be on CUDA device"
    
    # Create output tensor with same shape and type as input
    dst = torch.empty_like(src, device='cuda')
    
    # Calculate grid size
    n_elements = src.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    mul_kernel[grid](
        src_ptr=src.data_ptr(),
        dst_ptr=dst.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return dst

# Test the implementation
if __name__ == "__main__":
    # Create test tensor
    src = torch.tensor([8323072], dtype=torch.float32, device='cuda')
    
    # Run kernel
    dst = launch_mul_kernel(src)
    
    # Verify results
    expected = src * (2.0 ** (127 - 15))
    torch.testing.assert_close(dst, expected)
    
    print(f"Source: {src}")
    print(f"Result: {dst}")
    print(f"Expected: {expected}")
