import torch
import triton
import triton.language as tl

device = 'cuda:0'

# Triton kernel to compute the Airy function Ai
@triton.jit
def airy_ai_kernel(
    input_ptr,
    output_ptr,
    size,
    BLOCK_SIZE: tl.constexpr
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    # Compute the range of elements this program instance will process
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    # Load the input elements
    input_elements = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Compute the Airy function Ai for each element
    output_elements = tl.math.airy_ai(input_elements)

    # Store the results
    tl.store(output_ptr + offsets, output_elements, mask=mask)

# Wrapper function to compute the Airy function Ai
def airy_ai(input, *, out=None):
    # Ensure the input tensor is on the GPU
    assert input.is_cuda, 'Input tensor must be on GPU'
    
    # Prepare the output tensor
    if out is None:
        out = torch.empty_like(input, device=device)
    else:
        assert out.is_cuda, 'Output tensor must be on GPU'
        assert out.shape == input.shape, 'Output tensor must have the same shape as input tensor'
    
    # Get the size of the input tensor
    size = input.numel()
    
    # Determine the block size
    BLOCK_SIZE = triton.next_power_of_2(min(1024, size))
    
    # Launch the Triton kernel
    grid = (triton.cdiv(size, BLOCK_SIZE),)
    airy_ai_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        size=size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
