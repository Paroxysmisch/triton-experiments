import triton
import triton.language as tl

@triton.jit
def cos_signbit_kernel(input_ptr, output_cos_ptr, output_signbit_ptr, n_elements):
    # Get the index of the current element
    idx = tl.program_id(0) * tl.block_size(0) + tl.arange(0, tl.block_size(0))
    mask = idx < n_elements  # Mask to avoid out-of-bounds access

    # Load input tensor
    input_tensor = tl.load(input_ptr + idx)

    # Compute cosine
    cos_result = tl.cos(input_tensor)

    # Store cosine result
    tl.store(output_cos_ptr + idx, cos_result, mask=mask)

    # Compute sign bit
    signbit_result = tl.signbit(cos_result)

    # Store sign bit result
    tl.store(output_signbit_ptr + idx, signbit_result, mask=mask)

from typing import Tuple
import torch
import triton
import triton.language as tl

def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Ensure input is a 1D tensor
    assert input.ndim == 1, "Input tensor must be 1D"
    
    n_elements = input.numel()
    
    # Allocate output tensors
    output_cos = torch.empty_like(input)
    output_signbit = torch.empty(n_elements, dtype=torch.bool)

    # Launch the Triton kernel
    grid = (n_elements + 1023) // 1024  # Number of blocks
    cos_signbit_kernel[grid](input_ptr=input.data_ptr(), 
                              output_cos_ptr=output_cos.data_ptr(), 
                              output_signbit_ptr=output_signbit.data_ptr(), 
                              n_elements=n_elements)

    return output_cos, output_signbit
