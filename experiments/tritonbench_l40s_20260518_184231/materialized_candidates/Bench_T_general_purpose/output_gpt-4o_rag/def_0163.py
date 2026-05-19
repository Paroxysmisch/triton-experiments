import torch
import triton
import triton.language as tl
import math
from typing import Tuple

# Kernel function: Computes the cosine of each element and determines the sign bit.
@triton.jit
def cos_signbit_kernel(a, cos_result, sign_bit, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block and thread
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Determine which elements are valid within the block
    mask = offset < n_elements
    # Load elements from tensor `a` into `a_value`
    a_value = tl.load(a + offset, mask=mask)
    # Compute the cosine of each element in `a_value`
    cos_value = tl.cos(a_value.to(tl.float32))
    # Store the cosine result
    tl.store(cos_result + offset, cos_value, mask=mask)
    # Determine the sign bit of each cosine result
    sign_bit_value = cos_value < 0
    # Store the sign bit result
    tl.store(sign_bit + offset, sign_bit_value, mask=mask)

# Wrapper function: Computes the cosine and sign bit for each element in the input tensor.
def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Prepare output tensors `cos_result` and `sign_bit`
    cos_result = torch.empty_like(input)
    sign_bit = torch.empty_like(input, dtype=torch.bool)
    # Determine the total number of elements in the input tensor
    n_elements = input.numel()
    # Calculate the optimal block size
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    # Determine the grid size needed to cover all elements
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch the Triton kernel
    cos_signbit_kernel[(grid_size, 1, 1)](input, cos_result, sign_bit, n_elements, block_size)
    return cos_result, sign_bit
