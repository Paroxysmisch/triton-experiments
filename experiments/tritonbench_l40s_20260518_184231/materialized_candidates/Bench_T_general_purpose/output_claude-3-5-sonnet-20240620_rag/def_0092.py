import triton
import triton.language as tl
import torch

@triton.jit
def tensordot_rsqrt_kernel(
    output_ptr,
    input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the offset for this program instance
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for bounds checking
    mask = offsets < n_elements
    
    # Load the input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute reciprocal square root
    result = tl.where(x > 0, 1.0 / tl.sqrt(x), 0.0)
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)

def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    # Compute the tensordot product first
    tensordot_result = torch.tensordot(a, b, dims)
    
    # Prepare for Triton kernel
    output = torch.empty_like(tensordot_result)
    n_elements = tensordot_result.numel()
    
    # Define block size and grid
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 512))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    tensordot_rsqrt_kernel[grid](
        output,
        tensordot_result,
        n_elements,
        BLOCK_SIZE,
    )
    
    return output
