import triton
import triton.language as tl
import torch
import math

# Kernel for tensor other_mul and tensor other_sub
@triton.jit
def mul_sub_tensor_tensor(
    input_ptr,
    other_mul_ptr,
    other_sub_ptr,
    alpha: tl.constexpr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_val = tl.load(input_ptr + offsets, mask=mask)
    mul_val = tl.load(other_mul_ptr + offsets, mask=mask)
    sub_val = tl.load(other_sub_ptr + offsets, mask=mask)
    res = input_val * mul_val - alpha * sub_val
    tl.store(output_ptr + offsets, res, mask=mask)

# Kernel for tensor other_mul and scalar other_sub
@triton.jit
def mul_sub_tensor_scalar(
    input_ptr,
    other_mul_ptr,
    other_sub: tl.constexpr,
    alpha: tl.constexpr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_val = tl.load(input_ptr + offsets, mask=mask)
    mul_val = tl.load(other_mul_ptr + offsets, mask=mask)
    res = input_val * mul_val - alpha * other_sub
    tl.store(output_ptr + offsets, res, mask=mask)

# Kernel for scalar other_mul and tensor other_sub
@triton.jit
def mul_sub_scalar_tensor(
    input_ptr,
    other_mul: tl.constexpr,
    other_sub_ptr,
    alpha: tl.constexpr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_val = tl.load(input_ptr + offsets, mask=mask)
    sub_val = tl.load(other_sub_ptr + offsets, mask=mask)
    res = input_val * other_mul - alpha * sub_val
    tl.store(output_ptr + offsets, res, mask=mask)

# Kernel for scalar other_mul and scalar other_sub
@triton.jit
def mul_sub_scalar_scalar(
    input_ptr,
    other_mul: tl.constexpr,
    other_sub: tl.constexpr,
    alpha: tl.constexpr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_val = tl.load(input_ptr + offsets, mask=mask)
    res = input_val * other_mul - alpha * other_sub
    tl.store(output_ptr + offsets, res, mask=mask)

def mul_sub(input, other_mul, other_sub, alpha=1, out=None) -> torch.Tensor:
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"

    n_elements = input.numel()
    # Compute block and grid sizes
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)

    # Determine types of other_mul and other_sub
    is_other_mul_tensor = isinstance(other_mul, torch.Tensor)
    is_other_sub_tensor = isinstance(other_sub, torch.Tensor)

    # Check tensor shapes if applicable
    if is_other_mul_tensor:
        assert other_mul.shape == input.shape, "other_mul must match input shape"
    if is_other_sub_tensor:
        assert other_sub.shape == input.shape, "other_sub must match input shape"

    # Dispatch to the appropriate kernel
    if is_other_mul_tensor and is_other_sub_tensor:
        mul_sub_tensor_tensor[(grid_size, 1, 1)](
            input, other_mul, other_sub, alpha, out, n_elements, block_size
        )
    elif is_other_mul_tensor and not is_other_sub_tensor:
        mul_sub_tensor_scalar[(grid_size, 1, 1)](
            input, other_mul, other_sub, alpha, out, n_elements, block_size
        )
    elif not is_other_mul_tensor and is_other_sub_tensor:
        mul_sub_scalar_tensor[(grid_size, 1, 1)](
            input, other_mul, other_sub, alpha, out, n_elements, block_size
        )
    else:
        mul_sub_scalar_scalar[(grid_size, 1, 1)](
            input, other_mul, other_sub, alpha, out, n_elements, block_size
        )
    return out
