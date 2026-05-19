import torch
import triton
import triton.language as tl


@triton.jit
def _mul_relu_kernel(
    input_ptr, 
    other_ptr, 
    out_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.load(other_ptr + offsets, mask=mask)
    val = x * y
    val = tl.where(val > 0, val, 0.0)

    tl.store(out_ptr + offsets, val, mask=mask)


def mul_relu(input, other, inplace=False, out=None):
    if not input.is_cuda:
        raise ValueError("input must be a CUDA tensor.")

    if isinstance(other, (int, float)):
        other = torch.full_like(input, float(other)).cuda()
    else:
        if not other.is_cuda:
            raise ValueError("other must be a CUDA tensor or scalar.")

    if input.shape != other.shape:
        raise ValueError("input and other must have the same shape for this implementation.")

    if out is not None:
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor if provided.")
        if out.shape != input.shape:
            raise ValueError("out must have the same shape as input.")
    else:
        out = input if inplace else torch.empty_like(input)

    input_c = input.contiguous()
    other_c = other.contiguous()
    out_c = out.contiguous()

    n_elements = input.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _mul_relu_kernel[grid](input_c, other_c, out_c, n_elements, BLOCK_SIZE=1024)

    return out
