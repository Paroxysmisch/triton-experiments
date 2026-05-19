import torch
import triton
import triton.language as tl

device = 'cuda:0'

# Triton kernel for computing the error function
@triton.jit
def erf_kernel(
    input_ptr,
    output_ptr,
    size,
    BLOCK_SIZE: tl.constexpr
):
    # Get the current block index
    pid = tl.program_id(axis=0)
    # Calculate the range of elements this block will process
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    # Load the input elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Compute the error function using the approximation
    # erf(x) = 2 / sqrt(pi) * integral from 0 to x of exp(-t^2) dt
    # Using a polynomial approximation for erf(x)
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    t = 1.0 / (1.0 + p * tl.abs(x))
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * tl.exp(-x * x)
    y = tl.where(x >= 0, y, -y)

    # Store the results
    tl.store(output_ptr + offsets, y, mask=mask)

# Wrapper function for the erf operation
def erf(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Get the size of the input tensor
    size = input.numel()
    # Prepare the output tensor
    if out is None:
        out = torch.empty_like(input, device=device)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input tensor"
        assert out.is_cuda, "Output tensor must be on the same device as input tensor"

    # Ensure the input tensor is on the GPU
    assert input.is_cuda, "Input tensor must be on the GPU"

    # Determine the block size
    BLOCK_SIZE = triton.next_power_of_2(size)

    # Launch the Triton kernel
    erf_kernel[(size + BLOCK_SIZE - 1) // BLOCK_SIZE, ](
        input_ptr=input, output_ptr=out, size=size, BLOCK_SIZE=BLOCK_SIZE
    )

    return out
