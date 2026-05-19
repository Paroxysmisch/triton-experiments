import triton
import triton.language as tl

@triton.jit
def fused_masked_select_add_gelu_kernel(
    input_ptr, mask_ptr, other_ptr, output_ptr,
    alpha, N,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the block index
    block_idx = tl.program_id(0)
    
    # Create a block of indices
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load data from memory
    input_data = tl.load(input_ptr + offsets, mask=offsets < N, other=0.0)
    mask_data = tl.load(mask_ptr + offsets, mask=offsets < N, other=False)
    other_data = tl.load(other_ptr + offsets, mask=offsets < N, other=0.0)
    
    # Apply the mask
    selected_data = tl.where(mask_data, input_data, 0.0)
    
    # Add the scaled 'other' value
    result = selected_data + alpha * other_data
    
    # Apply GELU activation
    if approximate == 'tanh':
        cdf = 0.5 * (1.0 + tl.tanh((0.7978845608 * (result + 0.044715 * result * result * result))))
        gelu_result = result * cdf
    else:
        gelu_result = 0.5 * result * (1.0 + tl.erf(result / tl.sqrt(2.0)))
    
    # Store the result
    tl.store(output_ptr + offsets, gelu_result, mask=offsets < N)

import torch

def fused_masked_select_add_gelu(input, mask, other, *, alpha=1, approximate='none', out=None):
    # Ensure input is a torch tensor
    input = input.contiguous()
    mask = mask.contiguous()
    other = other.contiguous()

    # Get the number of elements
    N = input.numel()

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Define the block size
    BLOCK_SIZE = 1024  # You can tune this based on your hardware

    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    fused_masked_select_add_gelu_kernel[grid](
        input_ptr=input,
        mask_ptr=mask,
        other_ptr=other,
        output_ptr=out,
        alpha=alpha,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
