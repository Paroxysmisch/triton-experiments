import torch
import triton
import triton.language as tl
import math

device = 'cuda:0'

@triton.jit
def bessel_j1_kernel(input_ptr, output_ptr, n_elements: tl.constexpr):
    # Get the index for the current program
    idx = tl.program_id(axis=0)
    
    # Ensure we do not go out of bounds
    if idx >= n_elements:
        return

    # Load the input value
    x = tl.load(input_ptr + idx)

    # Compute the Bessel function of the first kind of order 1
    # Using the series expansion for Bessel function J1
    # J1(x) = (1/pi) * integral from 0 to pi of cos(x * sin(t)) * sin(t) dt
    # For simplicity, we will use a numerical approximation here
    result = (1 / math.pi) * tl.sin(x) * (tl.exp(x) - tl.exp(-x)) / (2 * x)

    # Store the result in the output tensor
    tl.store(output_ptr + idx, result)

def bessel_j1(input: torch.Tensor, *, out=None) -> torch.Tensor:
    # Prepare the output tensor
    if out is None:
        out = torch.empty_like(input)

    # Ensure both input and output tensors are on the GPU
    assert input.is_cuda and out.is_cuda, 'One of the tensors is not on GPU'

    n_elements = input.numel()
    
    # Launch the Triton kernel
    bessel_j1_kernel[(n_elements,)](
        input_ptr=input, output_ptr=out, n_elements=n_elements
    )

    return out
