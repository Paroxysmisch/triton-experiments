import triton
import triton.language as tl
import torch
import math

# ------------------------------------------------------------------------------
# Internal helper for rounding mode to integer code
def _rounding_mode_code(mode_str):
    # 0 -> no rounding, 1 -> floor, 2 -> trunc
    if mode_str == 'floor':
        return 1
    elif mode_str == 'trunc':
        return 2
    return 0

# ------------------------------------------------------------------------------
# Kernel: div_tensor_tensor_kernel
@triton.jit
def div_tensor_tensor_kernel(
    input_ptr, other_ptr, out_ptr,
    n_elements, rounding_code,
    BLOCK_SIZE: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    y = tl.load(other_ptr + offset, mask=mask)
    result = x / y

    # Rounding
    # rounding_code: 0 -> none, 1 -> floor, 2 -> trunc
    if rounding_code == 1:
        result = tl.floor(result)
    elif rounding_code == 2:
        # sign-preserving truncation
        # emulate "trunc" as floor for positive and ceil for negative
        positive_mask = result >= 0
        floored = tl.floor(result)
        ceiled = tl.ceil(result)
        result = tl.where(positive_mask, floored, ceiled)

    tl.store(out_ptr + offset, result, mask=mask)

# ------------------------------------------------------------------------------
# Kernel: div_tensor_scalar_kernel
@triton.jit
def div_tensor_scalar_kernel(
    input_ptr, val, out_ptr,
    n_elements, rounding_code,
    BLOCK_SIZE: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    result = x / val

    if rounding_code == 1:
        result = tl.floor(result)
    elif rounding_code == 2:
        positive_mask = result >= 0
        floored = tl.floor(result)
        ceiled = tl.ceil(result)
        result = tl.where(positive_mask, floored, ceiled)

    tl.store(out_ptr + offset, result, mask=mask)

# ------------------------------------------------------------------------------
# Kernel: div_scalar_tensor_kernel
@triton.jit
def div_scalar_tensor_kernel(
    val, other_ptr, out_ptr,
    n_elements, rounding_code,
    BLOCK_SIZE: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    y = tl.load(other_ptr + offset, mask=mask)
    result = val / y

    if rounding_code == 1:
        result = tl.floor(result)
    elif rounding_code == 2:
        positive_mask = result >= 0
        floored = tl.floor(result)
        ceiled = tl.ceil(result)
        result = tl.where(positive_mask, floored, ceiled)

    tl.store(out_ptr + offset, result, mask=mask)

# ------------------------------------------------------------------------------
# Kernel: div_scalar_scalar_kernel
@triton.jit
def div_scalar_scalar_kernel(
    valx, valy, out_ptr,
    n_elements, rounding_code,
    BLOCK_SIZE: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Single result for each element
    result = valx / valy
    if rounding_code == 1:
        result = tl.floor(result)
    elif rounding_code == 2:
        positive_mask = result >= 0
        floored = tl.floor(result)
        ceiled = tl.ceil(result)
        result = tl.where(positive_mask, floored, ceiled)

    tl.store(out_ptr + offset, result, mask=mask)

# ------------------------------------------------------------------------------
# Wrapper: div(input, other, *, rounding_mode=None, out=None) -> Tensor
def div(input, other, *, rounding_mode=None, out=None):
    # Handle scalar case for 'input'
    input_is_tensor = isinstance(input, torch.Tensor)
    other_is_tensor = isinstance(other, torch.Tensor)

    # Promote 'input' to default dtype if integer
    if input_is_tensor:
        if not (input.is_floating_point() or input.is_complex()):
            input = input.to(torch.get_default_dtype())

    # Promote 'other' to default dtype if it's a torch.Tensor of integer type
    if other_is_tensor:
        if not (other.is_floating_point() or other.is_complex()):
            other = other.to(torch.get_default_dtype())

    # Convert Python scalar 'other' to correct default dtype
    if not other_is_tensor and isinstance(other, int):
        other = float(other)

    # Convert Python scalar 'input' to correct default dtype
    if not input_is_tensor and isinstance(input, int):
        input = float(input)

    # For broadcasting, expand to common shape if both are tensors
    if input_is_tensor and other_is_tensor:
        shape = torch.broadcast_shapes(input.shape, other.shape)
        input_expanded = input.expand(shape).contiguous()
        other_expanded = other.expand(shape).contiguous()
    elif input_is_tensor:
        input_expanded = input
        other_expanded = other  # scalar
        shape = input.shape
    elif other_is_tensor:
        input_expanded = input  # scalar
        other_expanded = other
        shape = other.shape
    else:
        # both scalar
        shape = (1,)
        input_expanded = input
        other_expanded = other

    # Prepare output
    if out is None:
        if input_is_tensor:
            out = torch.empty_like(input_expanded, dtype=torch.get_default_dtype())
        elif other_is_tensor:
            out = torch.empty_like(other_expanded, dtype=torch.get_default_dtype())
        else:
            # both scalar => shape (1,)
            out = torch.empty(shape, dtype=torch.get_default_dtype())
    else:
        # Ensure out can hold the broadcasted shape
        if tuple(out.shape) != shape:
            raise ValueError("Output tensor has invalid shape.")
        if not out.is_contiguous():
            out = out.contiguous()

    # Flatten
    if isinstance(out, torch.Tensor):
        out_ = out.view(-1)
        n_elements = out_.numel()
    else:
        n_elements = 1

    rounding_code = _rounding_mode_code(rounding_mode)
    block_size = triton.next_power_of_2(min(n_elements, 1024))
    grid = ( (n_elements + block_size - 1) // block_size, )

    # Dispatch appropriate kernel
    if input_is_tensor and other_is_tensor:
        div_tensor_tensor_kernel[grid](
            input_expanded.view(-1), other_expanded.view(-1), out_,
            n_elements, rounding_code, BLOCK_SIZE=block_size
        )
    elif input_is_tensor and not other_is_tensor:
        div_tensor_scalar_kernel[grid](
            input_expanded.view(-1), other_expanded, out_,
            n_elements, rounding_code, BLOCK_SIZE=block_size
        )
    elif not input_is_tensor and other_is_tensor:
        div_scalar_tensor_kernel[grid](
            input_expanded, other_expanded.view(-1), out_,
            n_elements, rounding_code, BLOCK_SIZE=block_size
        )
    else:
        div_scalar_scalar_kernel[grid](
            input_expanded, other_expanded, out_,
            n_elements, rounding_code, BLOCK_SIZE=block_size
        )

    return out.view(shape)
