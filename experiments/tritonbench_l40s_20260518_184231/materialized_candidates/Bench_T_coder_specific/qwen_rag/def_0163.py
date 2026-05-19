import triton
import triton.language as tl
import torch
from typing import Tuple

# Kernel function: Computes the cosine of each element in the input tensor and determines the sign bit.
@triton.jit
def cos_signbit_kernel(a_ptr, b_ptr, sign_bit_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block and thread
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Determine which elements are valid within the block
    mask = offset < n_elements
    # Load elements from tensor `a` into `a_value`
    a_value = tl.load(a_ptr + offset, mask=mask)
    # Compute the cosine of each element in `a_value`
    cos_result = tl.cos(a_value.to(tl.float32))
    # Store the cosine result back to tensor `b`
    tl.store(b_ptr + offset, cos_result, mask=mask)
    # Compute the sign bit of each cosine result
    sign_bit = tl.signbit(cos_result)
    # Store the sign bit back to tensor `sign_bit_ptr`
    tl.store(sign_bit_ptr + offset, sign_bit, mask=mask)

# Wrapper function to invoke the Triton kernel and perform the computation
def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Prepare output tensor `B` for cosine results with the same shape and type as `input`
    B = torch.empty_like(input)
    # Prepare output tensor `sign_bit` for sign bits with the same shape and type as `input`
    sign_bit = torch.empty_like(input, dtype=torch.bool)
    # Determine the total number of elements in the input tensor `input`
    n_elements = input.numel()
    # Calculate the optimal block size
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    # Determine the grid size needed to cover all elements
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch the Triton kernel
    cos_signbit_kernel[(grid_size, 1, 1)](input, B, sign_bit, n_elements, block_size)
    return B, sign_bit

# Example usage
if __name__ == "__main__":
    input_tensor = torch.tensor([0.0, math.pi/4, math.pi/2, math.pi], dtype=torch.float32)
    cos_results, sign_bits = cos_signbit(input_tensor)
    print("Cosine Results:", cos_results)
    print("Sign Bits:", sign_bits)
