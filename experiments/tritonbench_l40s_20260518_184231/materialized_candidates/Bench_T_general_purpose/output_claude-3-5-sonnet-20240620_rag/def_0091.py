import triton
import triton.language as tl
import torch
import math

@triton.jit
def erfc_sqrt_kernel(
    input_ptr,
    erfc_out_ptr,
    sqrt_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate offset for parallel processing
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary checking
    mask = offset < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offset, mask=mask)
    
    # Convert to float32 for computation
    x = x.to(tl.float32)
    
    # Compute square root
    sqrt_result = tl.sqrt(x)
    
    # Compute erfc using approximation
    # Using Abramowitz and Stegun approximation for erfc
    t = 1.0 / (1.0 + 0.3275911 * x)
    erfc_result = t * (
        0.254829592 + 
        t * (-0.284496736 + 
        t * (1.421413741 + 
        t * (-1.453152027 + 
        t * 1.061405429)))
    )
    erfc_result = erfc_result * tl.exp(-x * x)
    
    # Store results
    tl.store(sqrt_out_ptr + offset, sqrt_result, mask=mask)
    tl.store(erfc_out_ptr + offset, erfc_result, mask=mask)

def erfc_sqrt(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the complementary error function (erfc) and square root of each element in the input tensor.
    
    Args:
        input: Input tensor
        
    Returns:
        Tuple containing (erfc_result, sqrt_result)
    """
    # Input validation
    assert input.is_cuda, "Input tensor must be on GPU"
    
    # Create output tensors
    erfc_output = torch.empty_like(input)
    sqrt_output = torch.empty_like(input)
    
    # Calculate launch grid parameters
    n_elements = input.numel()
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    erfc_sqrt_kernel[grid](
        input_ptr=input,
        erfc_out_ptr=erfc_output,
        sqrt_out_ptr=sqrt_output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return erfc_output, sqrt_output
