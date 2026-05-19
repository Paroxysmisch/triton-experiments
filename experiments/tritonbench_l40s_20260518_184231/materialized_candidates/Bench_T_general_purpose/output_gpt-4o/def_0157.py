import torch
import triton
import triton.language as tl
from typing import Tuple

# Triton kernel for computing the sign bit
@triton.jit
def signbit_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    # Compute the sign bit check
    result = input_vals < 0
    tl.store(output_ptr + offsets, result, mask=mask)

# Triton kernel for bitwise AND operation
@triton.jit
def bitwise_and_kernel(input_ptr, other_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    other_vals = tl.load(other_ptr + offsets, mask=mask)
    # Compute bitwise AND
    result = input_vals & other_vals
    tl.store(output_ptr + offsets, result, mask=mask)

# Wrapper function
def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input.dtype == torch.float32 or input.dtype == torch.float64, "Input tensor must be of floating point type."
    assert other.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.bool], "Other tensor must be of integral or boolean type."
    assert input.shape == other.shape, "Input and other tensors must have the same shape."

    n_elements = input.numel()
    signbit_result = torch.empty_like(input, dtype=torch.bool)
    bitwise_and_result = torch.empty_like(other)

    # Launch the Triton kernels
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    signbit_kernel[grid](input, signbit_result, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    bitwise_and_kernel[grid](input.to(other.dtype), other, bitwise_and_result, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return signbit_result, bitwise_and_result

# Example usage
a = torch.tensor([0.7, -1.2, 0., 2.3])
b = torch.tensor([1, 0, 1, 1], dtype=torch.int8)
signbit_result, bitwise_and_result = signbit_bitwise_and(a, b)
print(signbit_result)  # tensor([False, True, False, False])
print(bitwise_and_result)  # tensor([0, 0, 0, 0], dtype=torch.int8)
