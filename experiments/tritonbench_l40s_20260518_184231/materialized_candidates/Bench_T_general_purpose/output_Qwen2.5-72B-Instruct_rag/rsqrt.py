import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def rsqrt_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Get the current program ID
    pid = tl.program_id(axis=0)
    # Compute the start and end indices for the current block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the input elements with masking
    input_elements = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Compute the reciprocal of the square-root
    output_elements = 1.0 / tl.sqrt(input_elements)
    # Store the output elements with masking
    tl.store(output_ptr + offsets, output_elements, mask=mask)

def rsqrt(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Get the number of elements in the input tensor
    n_elements = input.numel()
    
    # Prepare the output tensor
    if out is None:
        out = torch.empty_like(input, device=device)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as the input tensor"
        assert out.is_cuda, "Output tensor must be on the same device as the input tensor"
    
    # Ensure the input tensor is on the GPU
    assert input.is_cuda, "Input tensor must be on the GPU"
    
    # Determine the block size
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    
    # Launch the Triton kernel
    rsqrt_kernel[(n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, ](
        input_ptr=input, output_ptr=out, n_elements=n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
