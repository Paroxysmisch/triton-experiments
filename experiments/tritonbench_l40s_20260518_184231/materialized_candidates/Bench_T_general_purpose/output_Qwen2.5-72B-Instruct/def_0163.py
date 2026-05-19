import triton
import triton.language as tl

@triton.jit
def cos_signbit_kernel(
    input_ptr,  # Pointer to the input tensor
    cos_result_ptr,  # Pointer to the output tensor for cosine results
    sign_bit_ptr,  # Pointer to the output tensor for sign bits
    n_elements,  # Number of elements in the input tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for parallel processing
):
    pid = tl.program_id(axis=0)  # Get the program ID
    block_start = pid * BLOCK_SIZE  # Calculate the start index for the block

    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Generate offsets for the block
    mask = offsets < n_elements  # Create a mask to handle cases where the block size is larger than the remaining elements

    input_block = tl.load(input_ptr + offsets, mask=mask)  # Load the input block
    cos_result_block = tl.cos(input_block)  # Compute the cosine of the input block

    # Determine the sign bit for each cosine result
    sign_bit_block = cos_result_block < 0

    # Store the results
    tl.store(cos_result_ptr + offsets, cos_result_block, mask=mask)
    tl.store(sign_bit_ptr + offsets, sign_bit_block, mask=mask)

import torch
from torch import Tensor
from typing import Tuple

def cos_signbit(input: Tensor) -> Tuple[Tensor, Tensor]:
    # Ensure the input tensor is on the same device as the Triton kernel
    device = input.device
    n_elements = input.numel()

    # Allocate output tensors
    cos_result = torch.empty_like(input, device=device)
    sign_bit = torch.empty_like(input, dtype=torch.bool, device=device)

    # Define the grid and block sizes
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    cos_signbit_kernel[grid, BLOCK_SIZE](
        input, cos_result, sign_bit, n_elements, BLOCK_SIZE
    )

    return cos_result, sign_bit
