import torch
import triton
import triton.language as tl

@triton.jit
def add_gelu_kernel(
    input_ptr, other_ptr, output_ptr,
    input_stride, other_stride, output_stride,
    n_elements,
    approximate_flag,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input = tl.load(input_ptr + offsets * input_stride, mask=mask)
    other = tl.load(other_ptr + offsets * other_stride, mask=mask)

    added = input + other

    if approximate_flag == 0:
        # Exact GELU using erf
        cdf = 0.5 * (1.0 + tl.math.erf(added * 0.7071067811865475))
        gelu = added * cdf
    else:
        # Approximate tanh version
        x = added
        x_cubed = x * x * x
        inner = x + 0.044715 * x_cubed
        inner = inner * 0.7978845608028654
        tanh_inner = tl.math.tanh(inner)
        gelu = 0.5 * x * (1.0 + tanh_inner)

    tl.store(output_ptr + offsets * output_stride, gelu, mask=mask)

def add_gelu(input, other, alpha=1, approximate='none', out=None):
    if approximate not in ['none', 'tanh']:
        raise ValueError("approximate must be 'none' or 'tanh'")
    approximate_flag = 0 if approximate == 'none' else 1

    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    scaled_other = other * alpha

    broadcasted_input, broadcasted_other = torch.broadcast_tensors(input, scaled_other)

    if out is None:
        out = torch.empty_like(broadcasted_input)
    else:
        if out.shape != broadcasted_input.shape:
            raise ValueError("Output tensor has incorrect shape")
        if out.device != broadcasted_input.device or out.dtype != broadcasted_input.dtype:
            raise ValueError("Output tensor device or dtype does not match input")

    input_flat = broadcasted_input.contiguous().view(-1)
    other_flat = broadcasted_other.contiguous().view(-1)
    output_flat = out.view(-1)

    n_elements = input_flat.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_gelu_kernel[grid](
        input_flat, other_flat, output_flat,
        input_flat.stride(0), other_flat.stride(0), output_flat.stride(0),
        n_elements,
        approximate_flag,
        BLOCK_SIZE=1024,
    )

    return out
