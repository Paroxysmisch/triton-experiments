import triton
import triton.language as tl

@triton.jit
def signbit_bitwise_and_kernel(
    input_ptr, other_ptr, signbit_out_ptr, bitwise_and_out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask)
    other_vals = tl.load(other_ptr + offsets, mask=mask)

    # Compute signbit
    signbit_vals = input_vals < 0
    tl.store(signbit_out_ptr + offsets, signbit_vals, mask=mask)

    # Compute bitwise AND
    bitwise_and_vals = tl.bitwise_and(input_vals, other_vals)
    tl.store(bitwise_and_out_ptr + offsets, bitwise_and_vals, mask=mask)

import torch
import triton
from triton.runtime import jit

def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.bool], "input must be of integral or boolean type"
    assert other.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.bool], "other must be of integral or boolean type"
    assert input.shape == other.shape, "input and other must have the same shape"

    # Convert input and other to Triton-compatible tensors
    input_t = triton.to_triton(input, device=input.device)
    other_t = triton.to_triton(other, device=other.device)

    # Allocate output tensors
    signbit_out = torch.empty_like(input, dtype=torch.bool)
    bitwise_and_out = torch.empty_like(input, dtype=input.dtype)

    # Convert output tensors to Triton-compatible tensors
    signbit_out_t = triton.to_triton(signbit_out, device=signbit_out.device)
    bitwise_and_out_t = triton.to_triton(bitwise_and_out, device=bitwise_and_out.device)

    # Launch the Triton kernel
    grid = (triton.cdiv(input.numel(), 1024),)
    signbit_bitwise_and_kernel[grid](
        input_t, other_t, signbit_out_t, bitwise_and_out_t, input.numel(), BLOCK_SIZE=1024
    )

    # Convert the results back to PyTorch tensors
    signbit_out = triton.from_triton(signbit_out_t, device=signbit_out.device)
    bitwise_and_out = triton.from_triton(bitwise_and_out_t, device=bitwise_and_out.device)

    return signbit_out, bitwise_and_out

import torch

a = torch.tensor([0.7, -1.2, 0., 2.3])
b = torch.tensor([1, 0, 1, 1], dtype=torch.int8)

signbit_result, bitwise_and_result = signbit_bitwise_and(a, b)

print("signbit_result:", signbit_result)
print("bitwise_and_result:", bitwise_and_result)
