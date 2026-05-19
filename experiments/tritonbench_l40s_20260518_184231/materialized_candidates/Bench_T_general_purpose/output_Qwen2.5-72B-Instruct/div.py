import triton
import triton.language as tl

@triton.jit
def div_kernel(
    input_ptr,  # Pointer to the input tensor
    other_ptr,  # Pointer to the other tensor or scalar
    out_ptr,    # Pointer to the output tensor
    input_n_elements,  # Number of elements in the input tensor
    other_n_elements,  # Number of elements in the other tensor or 1 for scalar
    input_strides,     # Strides of the input tensor
    other_strides,     # Strides of the other tensor or 0 for scalar
    out_strides,       # Strides of the output tensor
    input_dtype,       # Data type of the input tensor
    other_dtype,       # Data type of the other tensor or scalar
    out_dtype,         # Data type of the output tensor
    rounding_mode,     # Rounding mode
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_n_elements

    input_offsets = input_strides * offsets
    other_offsets = other_strides * (offsets % other_n_elements)
    out_offsets = out_strides * offsets

    input_vals = tl.load(input_ptr + input_offsets, mask=mask)
    other_vals = tl.load(other_ptr + other_offsets, mask=mask)

    if rounding_mode == 0:
        out_vals = input_vals / other_vals
    elif rounding_mode == 1:
        out_vals = tl.math.floor(input_vals / other_vals)
    elif rounding_mode == 2:
        out_vals = tl.math.ceil(input_vals / other_vals)
    else:
        out_vals = input_vals / other_vals

    tl.store(out_ptr + out_offsets, out_vals, mask=mask)

import torch
import triton
import triton.language as tl

def div(input, other, *, rounding_mode=None, out=None):
    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input, dtype=input.dtype)

    # Determine the data types and strides
    input_dtype = input.dtype
    other_dtype = other.dtype if isinstance(other, torch.Tensor) else input_dtype
    out_dtype = out.dtype

    input_n_elements = input.numel()
    other_n_elements = other.numel() if isinstance(other, torch.Tensor) else 1

    input_strides = input.stride()
    other_strides = other.stride() if isinstance(other, torch.Tensor) else 0
    out_strides = out.stride()

    # Determine the rounding mode
    rounding_mode_map = {
        None: 0,
        'floor': 1,
        'ceil': 2
    }
    rounding_mode = rounding_mode_map[rounding_mode]

    # Launch the Triton kernel
    grid = (triton.cdiv(input_n_elements, BLOCK_SIZE),)
    div_kernel[grid](
        input_ptr=input.data_ptr(),
        other_ptr=other.data_ptr() if isinstance(other, torch.Tensor) else other,
        out_ptr=out.data_ptr(),
        input_n_elements=input_n_elements,
        other_n_elements=other_n_elements,
        input_strides=input_strides,
        other_strides=other_strides,
        out_strides=out_strides,
        input_dtype=input_dtype,
        other_dtype=other_dtype,
        out_dtype=out_dtype,
        rounding_mode=rounding_mode,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
