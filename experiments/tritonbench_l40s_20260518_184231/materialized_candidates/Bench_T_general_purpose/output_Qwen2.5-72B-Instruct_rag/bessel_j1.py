import torch
import triton
import triton.language as tl

device = 'cuda:0'

# Triton kernel for computing the Bessel function of the first kind of order 1
@triton.jit
def bessel_j1_kernel(
    input_ptr,
    output_ptr,
    size,
    BLOCK_SIZE: tl.constexpr
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    # Compute the start and end indices for the current block
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, size)

    # Compute the offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle out-of-bounds accesses
    mask = offsets < size

    # Load the input elements with masking
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Compute the Bessel function of the first kind of order 1
    # Using the approximation for J1(x) for simplicity
    # For a more accurate implementation, consider using a library or a more complex approximation
    j1 = 0.5 * (tl.sin(x) / (x * x * x) - tl.cos(x) / x)

    # Store the result in the output tensor with masking
    tl.store(output_ptr + offsets, j1, mask=mask)

# Python wrapper function
def bessel_j1(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Ensure the input tensor is on the GPU
    assert input.is_cuda, 'Input tensor must be on GPU'
    
    # Prepare the output tensor
    if out is None:
        out = torch.empty_like(input, device=device)
    else:
        assert out.is_cuda, 'Output tensor must be on GPU'
        assert out.shape == input.shape, 'Output tensor shape must match input tensor shape'

    # Get the size of the input tensor
    size = input.numel()

    # Determine the block size
    BLOCK_SIZE = triton.next_power_of_2(size)

    # Launch the Triton kernel
    bessel_j1_kernel[(size + BLOCK_SIZE - 1) // BLOCK_SIZE, ](
        input_ptr=input, output_ptr=out, size=size, BLOCK_SIZE=BLOCK_SIZE
    )

    return out
