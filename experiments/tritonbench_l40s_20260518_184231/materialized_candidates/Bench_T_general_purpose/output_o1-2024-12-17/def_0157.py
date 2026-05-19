import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def _signbit_kernel(
    input_ptr,  # float input
    output_ptr,  # bool/int8 output
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load float values
    x = tl.load(input_ptr + offsets, mask=mask)

    # Reinterpret bits as int32 to catch negative zero
    bits = tl.bitcast(x, tl.int32)
    sign = bits & 0x80000000
    # True if sign bit set, else False
    signbit_res = sign != 0

    tl.store(output_ptr + offsets, signbit_res.to(tl.int8), mask=mask)

@triton.jit
def _bitwise_and_kernel(
    input_ptr,  # integral/boolean
    other_ptr,  # integral/boolean
    output_ptr, # result
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(input_ptr + offsets, mask=mask)
    b = tl.load(other_ptr + offsets, mask=mask)
    c = a & b
    tl.store(output_ptr + offsets, c, mask=mask)

def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes two results:
      1) signbit_result: For each element in 'input', returns True if sign bit is set (including -0).
      2) bitwise_and_result: Computes the bitwise AND of 'input' and 'other', where both must be
         integral or boolean for this operation. If 'input' is floating point, it is cast to 'other.dtype'.

    Args:
        input (Tensor): The input tensor (can be float, integral, or bool).
        other (Tensor): The second tensor, must be integral or bool.

    Returns:
        (Tensor, Tensor): (signbit_result, bitwise_and_result)
    """
    # 1) signbit_result
    # Prepare output tensor (bool is stored as int8 in Triton)
    signbit_result = torch.empty_like(input, dtype=torch.int8)

    # Launch _signbit_kernel
    n_elements = input.numel()
    grid = lambda meta: ( (n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    _signbit_kernel[grid](
        input_ptr=input.contiguous().view(-1).cuda(),
        output_ptr=signbit_result.view(-1).cuda(),
        n_elements=n_elements,
        BLOCK_SIZE=1024
    )
    signbit_result = signbit_result.bool()

    # 2) bitwise_and_result
    # For bitwise_and, 'input' must be cast to match 'other' if needed
    if input.dtype != other.dtype or not (other.dtype.is_integer or other.dtype == torch.bool):
        # cast input to 'other' dtype if it's not already integral/boolean
        input_cast = input.to(other.dtype)
    else:
        input_cast = input

    bitwise_and_result = torch.empty_like(other)

    # Launch _bitwise_and_kernel
    n_elements_and = min(input_cast.numel(), other.numel())
    grid_and = lambda meta: (
        (n_elements_and + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],
    )
    _bitwise_and_kernel[grid_and](
        input_ptr=input_cast.contiguous().view(-1).cuda(),
        other_ptr=other.contiguous().view(-1).cuda(),
        output_ptr=bitwise_and_result.view(-1).cuda(),
        n_elements=n_elements_and,
        BLOCK_SIZE=1024
    )

    return signbit_result, bitwise_and_result.view(other.shape)
