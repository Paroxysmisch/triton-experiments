import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def erf_kernel(
        input_ptr,
        output_ptr,
        N,
        BLOCK_SIZE: tl.constexpr
):
    # Get the program index
    pid = tl.program_id(axis=0)
    # Calculate the offsets for the current block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask for out-of-bounds accesses
    mask = offsets < N

    # Load the input elements with masking
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Constants for the error function approximation
    a1, a2, a3, a4, a5, p = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911

    # Compute the sign of x
    sign = tl.where(x >= 0, 1.0, -1.0)
    abs_x = tl.abs(x)
    
    # Compute the approximation of erf
    t = 1.0 / (1.0 + p * abs_x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * tl.exp(-abs_x * abs_x)
    
    # Apply the sign to get the final result
    result = sign * y

    # Store the result with masking
    tl.store(output_ptr + offsets, result, mask=mask)


def erf(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Get the number of elements in the input tensor
    N = input.numel()

    # Prepare the output tensor
    if out is None:
        out = torch.empty_like(input)

    # Ensure both input and output tensors are on the GPU
    assert input.is_cuda and out.is_cuda, 'Both tensors must be on GPU'

    # Determine the block size for the kernel
    BLOCK_SIZE = 1024  # You can adjust this based on your needs

    # Launch the Triton kernel
    erf_kernel[(N + BLOCK_SIZE - 1) // BLOCK_SIZE](
        input_ptr=input, output_ptr=out, N=N, BLOCK_SIZE=BLOCK_SIZE
    )

    return out
