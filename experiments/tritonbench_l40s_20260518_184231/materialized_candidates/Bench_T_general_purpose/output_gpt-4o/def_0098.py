import triton
import triton.language as tl

@triton.jit
def sub_gelu_kernel(
    input_ptr, other_ptr, output_ptr,
    alpha, n_elements, approximate,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    input_data = tl.load(input_ptr + offsets, mask=mask)
    other_data = tl.load(other_ptr + offsets, mask=mask)

    # Perform subtraction and scaling
    result = input_data - alpha * other_data

    # Apply GELU
    if approximate == 0:  # exact
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
        cdf = 0.5 * (1.0 + tl.erf(result * sqrt_2_over_pi))
        gelu_result = result * cdf
    else:  # approximate using tanh
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
        coeff = 0.044715
        tanh_arg = sqrt_2_over_pi * (result + coeff * result * result * result)
        gelu_result = 0.5 * result * (1.0 + tl.tanh(tanh_arg))

    # Store the result
    tl.store(output_ptr + offsets, gelu_result, mask=mask)

import torch

def sub_gelu(input, other, alpha=1, approximate='none', out=None):
    assert isinstance(input, torch.Tensor), "input must be a torch.Tensor"
    
    if isinstance(other, (int, float)):
        other = torch.full_like(input, other)
    elif isinstance(other, torch.Tensor):
        assert input.shape == other.shape, "input and other must have the same shape"
    else:
        raise TypeError("other must be a Tensor or a Number")

    if out is None:
        out = torch.empty_like(input)

    n_elements = input.numel()
    BLOCK_SIZE = 1024  # You can adjust this based on your hardware

    approximate_flag = 0 if approximate == 'none' else 1

    sub_gelu_kernel[(n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE](
        input_ptr=input,
        other_ptr=other,
        output_ptr=out,
        alpha=alpha,
        n_elements=n_elements,
        approximate=approximate_flag,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
