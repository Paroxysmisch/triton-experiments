import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def tanh_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Get the block id
    pid = tl.program_id(axis=0)
    # Compute the start and end indices for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the input elements
    input_elements = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Compute the hyperbolic tangent
    output_elements = tl.tanh(input_elements)
    # Store the results
    tl.store(output_ptr + offsets, output_elements, mask=mask)

def tanh(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Ensure the input tensor is on the GPU
    assert input.is_cuda, 'Input tensor must be on GPU'
    
    # Prepare the output tensor
    if out is None:
        out = torch.empty_like(input, device=device)
    else:
        assert out.is_cuda, 'Output tensor must be on GPU'
        assert out.shape == input.shape, 'Output tensor shape must match input tensor shape'
    
    # Get the number of elements in the input tensor
    n_elements = input.numel()
    
    # Determine the block size
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    
    # Launch the Triton kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    tanh_kernel[grid](
        input_ptr=input, output_ptr=out, n_elements=n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
