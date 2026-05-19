import triton
import triton.language as tl
import torch
import math

@triton.jit
def div_tensor_tensor_kernel(
    input_ptr,
    other_ptr,
    out_ptr,
    n_elements,
    rounding_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_val = tl.load(input_ptr + offsets, mask=mask)
    other_val = tl.load(other_ptr + offsets, mask=mask)
    output_val = input_val / other_val

    if rounding_mode == 1:  # trunc
        output_val = tl.libdevice.trunc(output_val)
    elif rounding_mode == 2:  # floor
        output_val = tl.libdevice.floor(output_val)

    tl.store(out_ptr + offsets, output_val, mask=mask)

@triton.jit
def div_tensor_scalar_kernel(
    input_ptr,
    other_scalar,
    out_ptr,
    n_elements,
    rounding_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_val = tl.load(input_ptr + offsets, mask=mask)
    output_val = input_val / other_scalar

    if rounding_mode == 1:  # trunc
        output_val = tl.libdevice.trunc(output_val)
    elif rounding_mode == 2:  # floor
        output_val = tl.libdevice.floor(output_val)

    tl.store(out_ptr + offsets, output_val, mask=mask)

def div(input, other, *, rounding_mode=None, out=None):
    # Compute output dtype with type promotion
    if rounding_mode is None:
        dtype = torch.result_type(input, other)
        if not dtype.is_floating_point and not dtype.is_complex:
            dtype = torch.get_default_dtype()
    else:
        dtype = torch.result_type(input, other)
        if dtype.is_complex:
            raise ValueError("rounding_mode is not supported for complex inputs")
        if not dtype.is_floating_point:
            dtype = torch.result_type(input, other, torch.float32)

    # Handle broadcasting
    if isinstance(other, torch.Tensor):
        broadcast_shape = torch.broadcast_shapes(input.shape, other.shape)
        input_expanded = input.expand(broadcast_shape)
        other_expanded = other.expand(broadcast_shape)
    else:
        broadcast_shape = input.shape
        input_expanded = input
        other_expanded = other

    # Ensure contiguous tensors
    input_contig = input_expanded.contiguous()
    if isinstance(other, torch.Tensor):
        other_contig = other_expanded.contiguous()
    else:
        other_contig = other_expanded

    # Create output tensor
    if out is None:
        out = torch.empty(broadcast_shape, dtype=dtype, device=input.device)
    else:
        if out.shape != broadcast_shape:
            raise ValueError("out shape does not match broadcasted shape")
        if out.dtype != dtype:
            raise ValueError("out dtype does not match expected dtype")
    out_contig = out.contiguous()

    n_elements = input_contig.numel()
    if n_elements == 0:
        return out  # handle empty tensors

    block_size = 1024
    grid_size = (triton.cdiv(n_elements, block_size),)

    # Convert rounding mode to kernel parameter
    rounding_mode_kernel = 0
    if rounding_mode == 'trunc':
        rounding_mode_kernel = 1
    elif rounding_mode == 'floor':
        rounding_mode_kernel = 2

    # Launch appropriate kernel
    if isinstance(other, torch.Tensor):
        div_tensor_tensor_kernel[grid_size](
            input_contig, other_contig, out_contig, n_elements,
            rounding_mode_kernel, BLOCK_SIZE=block_size
        )
    else:
        # Convert scalar to the correct dtype
        if isinstance(other_contig, torch.Tensor):
            other_scalar = other_contig.item()
        else:
            other_scalar = other_contig
        other_scalar = torch.tensor(other_scalar, dtype=dtype).item()

        div_tensor_scalar_kernel[grid_size](
            input_contig, other_scalar, out_contig, n_elements,
            rounding_mode_kernel, BLOCK_SIZE=block_size
        )

    return out_contig
