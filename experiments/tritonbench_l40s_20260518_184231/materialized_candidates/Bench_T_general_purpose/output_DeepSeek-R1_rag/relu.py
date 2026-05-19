import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_data = tl.load(input_ptr + offsets, mask=mask)
    output_data = tl.where(input_data > 0, input_data, 0.0)
    tl.store(output_ptr + offsets, output_data, mask=mask)

def relu(input, inplace=False):
    if inplace:
        if not input.is_contiguous():
            raise RuntimeError("Inplace ReLU requires contiguous input tensor.")
        output = input
    else:
        if input.is_contiguous():
            output = torch.empty_like(input)
        else:
            input = input.contiguous()
            output = torch.empty_like(input)
    
    n_elements = input.numel()
    if n_elements == 0:
        return output  # Handle empty tensor case
    
    # Tuned block size for optimal performance; adjust based on hardware
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    relu_kernel[grid](input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output
