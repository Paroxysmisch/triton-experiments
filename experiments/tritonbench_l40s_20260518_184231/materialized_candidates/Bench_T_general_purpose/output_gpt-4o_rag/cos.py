import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def cos_kernel(
        input_ptr,
        output_ptr,
        N,
        BLOCK_SIZE: tl.constexpr
):
    # Get the program index
    pid = tl.program_id(axis=0)
    # Calculate offsets for the current block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle out-of-bounds accesses
    mask = offsets < N

    # Load the elements from input_ptr with masking
    input_elements = tl.load(input_ptr + offsets, mask=mask)
    # Compute the cosine of the elements
    cos_elements = tl.cos(input_elements)
    # Store the result in output_ptr with masking
    tl.store(output_ptr + offsets, cos_elements, mask=mask)

def cos(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Ensure the input tensor is on the GPU
    assert input.is_cuda, 'Input tensor must be on GPU'

    # Determine the number of elements in the input tensor
    N = input.numel()

    # Prepare an output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Ensure the output tensor is on the GPU
    assert out.is_cuda, 'Output tensor must be on GPU'

    # Block size will be a power of 2, ideally to fit GPU architecture
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    # Launch the Triton kernel
    cos_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
