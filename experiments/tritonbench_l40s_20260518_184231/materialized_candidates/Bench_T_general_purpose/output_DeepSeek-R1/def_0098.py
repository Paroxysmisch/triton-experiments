import torch
import triton
import triton.language as tl

@triton.jit
def sub_gelu_kernel(
    input_ptr, other_ptr, output_ptr,
    input_stride, other_stride, output_stride,
    n_elements,
    approx_mode,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_offsets = offsets * input_stride
    other_offsets = offsets * other_stride

    input_vals = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    other_vals = tl.load(other_ptr + other_offsets, mask=mask, other=0.0)

    subtracted = input_vals - other_vals

    if approx_mode == 1:
        # Approximate GELU using tanh
        sqrt_2_over_pi = 0.7978845608
        approx = 0.044715
        x = subtracted
        x_cubed = x * x * x
        inner = sqrt_2_over_pi * (x + approx * x_cubed)
        tanh_inner = tl.math.tanh(inner)
        gelu = 0.5 * x * (1 + tanh_inner)
    else:
        # Exact GELU using erf
        x = subtracted
        erf = tl.math.erf(x * 0.7071067811865475)  # sqrt(0.5)
        gelu = 0.5 * x * (1 + erf)

    output_offsets = offsets * output_stride
    tl.store(output_ptr + output_offsets, gelu, mask=mask)

def sub_gelu(input, other, alpha=1, approximate='none', out=None):
    adjusted_other = torch.mul(other, alpha)

    try:
        broadcasted_shape = torch.broadcast_shapes(input.shape, adjusted_other.shape)
    except RuntimeError as e:
        raise RuntimeError("input and other are not broadcastable") from e

    input_expanded = input.expand(broadcasted_shape)
    adjusted_other_expanded = adjusted_other.expand(broadcasted_shape)

    input_flat = input_expanded.contiguous().view(-1)
    adjusted_other_flat = adjusted_other_expanded.contiguous().view(-1)

    if out is not None:
        if out.shape != broadcasted_shape:
            raise RuntimeError("out tensor has incorrect shape")
        out_flat = out.view(-1)
    else:
        out_flat = torch.empty_like(input_flat)

    n_elements = input_flat.numel()
    approx_mode = 1 if approximate == 'tanh' else 0

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    BLOCK_SIZE = 1024
    sub_gelu_kernel[grid](
        input_flat, adjusted_other_flat, out_flat,
        input_flat.stride(0), adjusted_other_flat.stride(0), out_flat.stride(0),
        n_elements,
        approx_mode,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out if out is not None else out_flat.view(broadcasted_shape)
