import torch
import triton
import triton.language as tl
import math

device = 'cuda:0'

@triton.jit
def erf_kernel(
        input_ptr,
        output_ptr,
        N,
        BLOCK_SIZE: tl.constexpr
):
    # Get the index for the current program
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N  # Create a mask for out-of-bounds accesses

    # Load input values
    x = tl.load(input_ptr + idx, mask=mask)

    # Compute the error function using the Taylor series expansion
    # for simplicity, we will use a basic approximation
    # Note: This is a simplified version; a more accurate implementation may be needed
    result = (2 / math.sqrt(math.pi)) * tl.sum(
        tl.exp(-x**2) * (1 - (x**2) / 3), axis=0
    )

    # Store the result in the output tensor
    tl.store(output_ptr + idx, result, mask=mask)

def erf(input: torch.Tensor, *, out=None) -> torch.Tensor:
    # Get the number of elements in the input tensor
    N = input.numel()
    # Prepare an output tensor on the same device
    if out is None:
        output = torch.empty_like(input)
    else:
        output = out

    # Ensure both input and output tensors are on the GPU
    assert input.is_cuda and output.is_cuda, 'One of the tensors is not on GPU'

    # Block size can be set to a reasonable value
    BLOCK_SIZE = 1024  # You can adjust this based on your needs
    print(f'Block size: {BLOCK_SIZE}, grid: {(N + BLOCK_SIZE - 1) // BLOCK_SIZE}')

    # Launch the Triton kernel
    erf_kernel[(N + BLOCK_SIZE - 1) // BLOCK_SIZE](  # Grid size
        input_ptr=input, output_ptr=output, N=N, BLOCK_SIZE=BLOCK_SIZE
    )

    return output
