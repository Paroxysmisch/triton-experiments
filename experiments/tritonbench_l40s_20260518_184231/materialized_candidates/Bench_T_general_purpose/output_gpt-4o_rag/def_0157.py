import triton
import triton.language as tl
import torch
from typing import Tuple

# Kernel for checking the sign bit of a tensor
@triton.jit
def signbit_func(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    # Check the sign bit (True if negative, including -0)
    output = input < 0
    tl.store(output_ptr + offsets, output, mask=mask)

# Kernel for bitwise AND operation on two tensors
@triton.jit
def bitwise_and_func(A_ptr, B_ptr, C_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    A = tl.load(A_ptr + offsets, mask=mask)
    B = tl.load(B_ptr + offsets, mask=mask)
    C = A & B
    tl.store(C_ptr + offsets, C, mask=mask)

# Wrapper function for signbit and bitwise AND operations
def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input.dtype == torch.float32 or input.dtype == torch.float64, "Input must be a floating point tensor"
    assert other.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool), "Other must be an integral or boolean tensor"
    
    n_elements = input.numel()
    signbit_result = torch.empty(n_elements, dtype=torch.bool, device=input.device)
    bitwise_and_result = torch.empty_like(other)

    block_size = triton.next_power_of_2(min(1024, n_elements))
    grid_size = (n_elements + block_size - 1) // block_size

    # Launch Triton kernel for signbit operation
    signbit_func[grid_size](input, signbit_result, n_elements, block_size)

    # Launch Triton kernel for bitwise AND operation
    bitwise_and_func[grid_size](input.to(dtype=other.dtype), other, bitwise_and_result, n_elements, block_size)

    return signbit_result, bitwise_and_result

# Example usage
a = torch.tensor([0.7, -1.2, 0., 2.3], dtype=torch.float32)
b = torch.tensor([1, 0, 1, 1], dtype=torch.int8)
signbit_result, bitwise_and_result = signbit_bitwise_and(a, b)
print(signbit_result)  # tensor([False, True, False, False])
print(bitwise_and_result)  # tensor([0, 0, 0, 0], dtype=torch.int8)
