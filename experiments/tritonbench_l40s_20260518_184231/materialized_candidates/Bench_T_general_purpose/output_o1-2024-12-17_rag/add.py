import triton
import triton.language as tl
import torch
import math


@triton.jit
def _add_kernel_tensor(
    input_ptr, other_ptr, out_ptr,
    alpha, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp = tl.load(input_ptr + offsets, mask=mask)
    oth = tl.load(other_ptr + offsets, mask=mask)
    res = inp + alpha * oth
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def _add_kernel_scalar(
    input_ptr, scalar: tl.constexpr, out_ptr,
    alpha, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp = tl.load(input_ptr + offsets, mask=mask)
    res = inp + alpha * scalar
    tl.store(out_ptr + offsets, res, mask=mask)


def add(input, other, *, alpha=1, out=None):
    # Broadcast to common shape and type-promote
    if isinstance(other, (int, float, complex, torch.scalar_tensor)):
        other_tensor = None
    else:
        other_tensor = other

    # Broadcast if other is a tensor
    if other_tensor is not None:
        broadcast_shape = torch.broadcast_shapes(input.shape, other_tensor.shape)
        input_bc = input.expand(broadcast_shape).to(torch.promote_types(input.dtype, other_tensor.dtype))
        other_bc = other_tensor.expand(broadcast_shape).to(input_bc.dtype)
    else:
        # For scalar, no need to broadcast
        input_bc = input
        other_bc = None

    # Prepare the output tensor
    if out is not None:
        out_bc = out
    else:
        if other_bc is not None:
            out_bc = torch.empty_like(input_bc)
        else:
            # input_bc shape is input's shape if no broadcasting is needed
            out_bc = torch.empty_like(input_bc)

    # Num elements
    n_elements = out_bc.numel()
    block_size = triton.next_power_of_2(min(n_elements, 1024))
    grid_size = (n_elements + block_size - 1) // block_size

    # Launch appropriate kernel
    if other_bc is not None:
        _add_kernel_tensor[(grid_size,)](
            input_bc, other_bc, out_bc,
            alpha,
            n_elements,
            BLOCK_SIZE=block_size
        )
    else:
        _add_kernel_scalar[(grid_size,)](
            input_bc, other,
            out_bc,
            alpha,
            n_elements,
            BLOCK_SIZE=block_size
        )

    return out_bc
