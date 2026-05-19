import triton
import triton.language as tl
import torch

@triton.jit
def isfinite_func_kernel_rank_1(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Number of elements in tensor
    BLOCK_SIZE: tl.constexpr,  # Size of parallel block processing
):
    # Calculate pid (program ID) and the block of elements to process
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements (handling edge cases)
    mask = offsets < n_elements
    
    # Load input data using block pointer
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Check if values are finite using Triton's built-in function
    result = tl.where(mask, tl.isfinite(x), 0)
    
    # Store results
    tl.store(output_ptr + offsets, result, mask=mask)

def isfinite_func_wrapper_rank_1(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to check if elements in a tensor are finite.
    
    Args:
        input_tensor: Input tensor to check
        
    Returns:
        torch.Tensor: Boolean tensor indicating which elements are finite
    """
    # Input validation
    assert input_tensor.dim() == 1, "Input tensor must be rank 1"
    
    # Get tensor properties
    n_elements = input_tensor.numel()
    
    # Create output tensor
    output = torch.empty_like(input_tensor, dtype=torch.bool, device=input_tensor.device)
    
    # Calculate optimal block size (multiple of 128)
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 2048))
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    isfinite_func_kernel_rank_1[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
