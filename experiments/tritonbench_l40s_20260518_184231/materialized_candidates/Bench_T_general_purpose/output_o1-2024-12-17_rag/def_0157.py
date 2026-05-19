import triton
import triton.language as tl
import torch
import math
from typing import Tuple

# ------------------------------------------------------------------------------
# Triton kernel to compute the signbit of float32 elements (True if negative, including -0).
# ------------------------------------------------------------------------------
@triton.jit
def signbit_kernel_f32(
    input_ptr,       # Pointer to input tensor (float32)
    output_ptr,      # Pointer to output tensor (int32 for signbit storage)
    n_elements,      # Number of elements
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load float32 values
    x = tl.load(input_ptr + offsets, mask=mask)

    # Bitcast to int32
    xi = tl.bitcast_to_int32(x)
    # Extract sign bit
    negative_bit = xi & 0x80000000
    # Convert to 1 if sign bit is set, else 0
    sign = tl.where(negative_bit != 0, 1, 0)

    # Store the int32 sign result
    tl.store(output_ptr + offsets, sign, mask=mask)

# ------------------------------------------------------------------------------
# Triton kernel for bitwise AND operation on two tensors (both integral/boolean).
# ------------------------------------------------------------------------------
@triton.jit
def bitwise_and_func_tensor(
    A_ptr, B_ptr, C_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    A = tl.load(A_ptr + offsets, mask=mask)
    B = tl.load(B_ptr + offsets, mask=mask)
    C = A & B
    tl.store(C_ptr + offsets, C, mask=mask)

# ------------------------------------------------------------------------------
# Triton kernel for bitwise AND operation on a tensor A and a scalar B.
# ------------------------------------------------------------------------------
@triton.jit
def bitwise_and_func_scalar(
    A_ptr,
    B: tl.constexpr,
    C_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    A = tl.load(A_ptr + offsets, mask=mask)
    C = A & B
    tl.store(C_ptr + offsets, C, mask=mask)

# ------------------------------------------------------------------------------
# Python wrapper to dispatch the appropriate bitwise AND Triton kernel.
# ------------------------------------------------------------------------------
def bitwise_and(A: torch.Tensor, B: torch.Tensor):
    # A, B are integral/boolean
    C = torch.empty_like(A)
    n_elements = C.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid = triton.cdiv(n_elements, block_size)

    # If B is a scalar
    if B.numel() == 1:
        bitwise_and_func_scalar[(grid,)](
            A, B.item(), C, n_elements, block_size
        )
    else:
        bitwise_and_func_tensor[(grid,)](
            A, B, C, n_elements, block_size
        )
    return C

# ------------------------------------------------------------------------------
# Python wrapper to compute signbit for float32 input using a Triton kernel.
# ------------------------------------------------------------------------------
def signbit_f32(input: torch.Tensor) -> torch.Tensor:
    # Create int32 buffer to store sign bits
    temp_sign = torch.empty(
        input.numel(),
        dtype=torch.int32,
        device=input.device
    )
    n_elements = input.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid = triton.cdiv(n_elements, block_size)

    signbit_kernel_f32[(grid,)](
        input, temp_sign, n_elements, block_size
    )

    # Convert stored int32 sign bits into bool
    return (temp_sign != 0).reshape(input.shape)

# ------------------------------------------------------------------------------
# Final requested function: signbit_bitwise_and
# ------------------------------------------------------------------------------
def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the sign bit check and bitwise AND operation on the input tensors.

    Args:
        input (Tensor): The input tensor (float32 expected for signbit operation).
        other (Tensor): The second tensor for bitwise AND, should be integral or boolean.
    
    Returns:
        (Tensor, Tensor): (signbit_result, bitwise_and_result).
                          signbit_result is a bool tensor indicating sign bit is set.
                          bitwise_and_result is the element-wise bitwise AND of input and other.
    """
    # Check if input is float32 for signbit
    if input.dtype not in (torch.float32,):
        raise TypeError("signbit_bitwise_and currently only supports torch.float32 for 'input'.")

    # Check if other is integral or boolean
    if not (other.dtype.is_floating_point is False or other.dtype == torch.bool):
        raise TypeError("The 'other' tensor must be of integral or boolean type for bitwise AND.")

    # 1. Compute signbit
    signbit_result = signbit_f32(input)

    # 2. Compute bitwise_and
    bitwise_and_result = bitwise_and(other, other)  # dummy initialization
    if other.numel() == 1:
        # If 'other' is effectively a scalar
        bitwise_and_result = bitwise_and(input.to(other.dtype), other)
    else:
        bitwise_and_result = bitwise_and(input.to(other.dtype), other)

    return signbit_result, bitwise_and_result
