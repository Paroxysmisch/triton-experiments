import triton
import triton.language as tl

@triton.jit
def relu_sqrt_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_block = tl.load(input_ptr + offsets, mask=mask)
    
    # Apply ReLU
    input_block = tl.where(input_block > 0, input_block, 0.0)
    
    # Apply square root
    output_block = tl.sqrt(input_block)
    
    tl.store(output_ptr + offsets, output_block, mask=mask)

import torch
import triton
import triton.language as tl

def relu_sqrt(input, inplace=False, out=None) -> torch.Tensor:
    # Ensure input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    
    # Ensure input is on the same device as the Triton kernel
    device = input.device
    if device.type != 'cuda':
        raise RuntimeError("input tensor must be on a CUDA device")
    
    # Determine the output tensor
    if inplace:
        if out is not None:
            raise ValueError("inplace and out cannot both be set")
        out = input
    else:
        if out is None:
            out = torch.empty_like(input, device=device)
        else:
            if out.shape != input.shape:
                raise ValueError("out tensor must have the same shape as input")
    
    # Launch the Triton kernel
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    relu_sqrt_kernel[grid](
        input, out, n_elements, BLOCK_SIZE=1024
    )
    
    return out
