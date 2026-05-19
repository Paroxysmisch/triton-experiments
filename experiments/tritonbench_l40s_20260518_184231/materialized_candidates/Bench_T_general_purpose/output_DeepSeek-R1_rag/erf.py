import torch
import triton
import triton.language as tl

@triton.jit
def erf_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Constants for the polynomial approximation of erf
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Load the input element
    x = tl.load(input_ptr + offsets, mask=mask)

    # Compute the sign and absolute value
    sign = tl.where(x < 0, -1.0, 1.0)
    abs_x = tl.abs(x)

    # Compute t and the polynomial terms
    t = 1.0 / (1.0 + p * abs_x)
    poly = a1 + t * (a2 + t * (a3 + t * a4))
    erfc = poly * t * tl.exp(-abs_x * abs_x)

    # Calculate the error function value and apply the sign
    erf_val = sign * (1.0 - erfc)

    # Store the result
    tl.store(output_ptr + offsets, erf_val, mask=mask)

def erf(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Ensure the input tensor is on CUDA
    assert input.is_cuda, "Input tensor must be on a CUDA device"

    # Create the output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_cuda, "Output tensor must be on a CUDA device"
        assert out.shape == input.shape, "Output tensor shape must match input"

    num_elements = input.numel()
    if num_elements == 0:
        return out  # Handle empty tensor case

    # Set block size to a reasonable value for element-wise operations
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)

    # Launch the kernel
    erf_kernel[grid](input, out, num_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out
