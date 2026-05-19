import triton
import triton.language as tl
import torch

@triton.jit
def isfinite_func_kernel_rank_1(
    input_ptr,  # Pointer to input tensor
    output_ptr, # Pointer to output tensor
    n_elements, # Number of elements in tensor
    BLOCK_SIZE: tl.constexpr,  # Size of block to process
):
    # Calculate the absolute position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Check if values are finite using Triton's built-in functions
    # A number is finite if it's not inf and not nan
    is_not_inf = ~tl.isinf(x)
    is_not_nan = ~tl.isnan(x)
    result = is_not_inf & is_not_nan
    
    # Store results
    tl.store(output_ptr + offsets, result, mask=mask)

def isfinite_func_wrapper_rank_1(input_tensor, output_tensor):
    """
    Wrapper function to check if elements in input tensor are finite
    
    Args:
        input_tensor: Input tensor (float32 or float64)
        output_tensor: Output tensor (bool)
    """
    # Get tensor properties
    n_elements = input_tensor.numel()
    
    # Calculate optimal block size (multiple of 128)
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 2048))
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    isfinite_func_kernel_rank_1[grid](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

# Example usage
def example_usage():
    # Create test tensor with some non-finite values
    x = torch.tensor([1.0, float('inf'), float('nan'), 2.0], 
                    dtype=torch.float32, 
                    device='cuda')
    
    # Create output tensor
    output = torch.empty_like(x, dtype=torch.bool, device='cuda')
    
    # Run kernel
    isfinite_func_wrapper_rank_1(x, output)
    
    return output
