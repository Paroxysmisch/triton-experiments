import torch
import triton
import triton.language as tl

@triton.jit
def signbit_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask, dtype=tl.float32)
    int_vals = input_vals.view(tl.int32)
    sign_bits = (int_vals >> 31) & 1
    tl.store(output_ptr + offsets, sign_bits.to(tl.int8), mask=mask)

@triton.jit
def bitwise_and_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask, dtype=tl.int8)
    other_vals = tl.load(other_ptr + offsets, mask=mask, dtype=tl.int8)
    output_vals = input_vals & other_vals
    tl.store(output_ptr + offsets, output_vals, mask=mask)

def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if other.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
        raise TypeError("other tensor must be of integral or boolean type.")
    
    if not input.is_cuda or not other.is_cuda:
        raise RuntimeError("Tensors must be on CUDA device")
    
    n_elements = input.numel()
    signbit_output = torch.empty_like(input, dtype=torch.bool)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    signbit_kernel[grid](input, signbit_output, n_elements, BLOCK_SIZE=1024)
    
    input_cast = input.to(other.dtype)
    if input_cast.dtype != other.dtype:
        raise RuntimeError("Failed to cast input to other's dtype")
    
    n_elements_and = input_cast.numel()
    bitwise_and_output = torch.empty_like(input_cast)
    bitwise_and_kernel[grid](input_cast, other, bitwise_and_output, n_elements_and, BLOCK_SIZE=1024)
    
    return signbit_output, bitwise_and_output
