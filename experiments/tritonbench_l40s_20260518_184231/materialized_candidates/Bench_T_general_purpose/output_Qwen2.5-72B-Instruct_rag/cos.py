import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def cos_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Get the current program id
    pid = tl.program_id(axis=0)
    # Compute the range of elements this program instance will process
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the input elements
    input_elements = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Compute the cosine of the input elements
    output_elements = tl.cos(input_elements)
    # Store the result in the output tensor
    tl.store(output_ptr + offsets, output_elements, mask=mask)

def cos(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Ensure the input tensor is on the GPU
    assert input.is_cuda, 'Input tensor must be on GPU'
    
    # If out is not provided, create a new tensor for the output
    if out is None:
        out = torch.empty_like(input, device=device)
    else:
        # Ensure the output tensor is on the GPU and has the same shape as the input
        assert out.is_cuda, 'Output tensor must be on GPU'
        assert out.shape == input.shape, 'Output tensor must have the same shape as input tensor'
    
    # Calculate the number of elements in the tensor
    n_elements = input.numel()
    # Determine the block size for parallel execution
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
    
    # Launch the Triton kernel
    cos_kernel[(n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, ](
        input_ptr=input, output_ptr=out, n_elements=n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
