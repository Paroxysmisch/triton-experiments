import torch
import triton
import triton.language as tl

@triton.jit
def pow_func_scalar_tensor_kernel_rank_1(
    output_ptr,    # Pointer to output tensor
    input_ptr,     # Pointer to input tensor
    scalar,        # Scalar power value
    n_elements,    # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Size of each block for processing
):
    # Calculate the absolute position of the block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create a block-specific offset for memory loading
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for boundary checking
    mask = offsets < n_elements
    
    # Load input data using the mask
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute power operation
    output = tl.pow(x, scalar)
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def pow_func_scalar_tensor_wrapper_rank_1(
    input_tensor: torch.Tensor,
    scalar: float,
) -> torch.Tensor:
    # Input validation
    assert input_tensor.dim() == 1, "Input tensor must be 1-dimensional"
    
    # Create output tensor with same shape and dtype
    output = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()
    
    # Determine optimal block size (power of 2)
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    pow_func_scalar_tensor_kernel_rank_1[grid](
        output,
        input_tensor,
        scalar,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,  # Can be tuned based on workload
    )
    
    return output

# Example usage
if __name__ == "__main__":
    # Create sample input
    x = torch.randn(1000000, device='cuda')
    scalar = 2.0
    
    # Run the power operation
    result = pow_func_scalar_tensor_wrapper_rank_1(x, scalar)
    
    # Verify result
    expected = torch.pow(x, scalar)
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)
    print("Verification passed!")
