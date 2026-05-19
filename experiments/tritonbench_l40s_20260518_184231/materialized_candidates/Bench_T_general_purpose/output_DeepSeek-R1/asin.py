import triton
import triton.language as tl
import torch

@triton.jit
def asin_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    # Check if input is within the valid range [-1, 1]
    condition = (input >= -1.0) & (input <= 1.0)
    # Compute arcsine and replace invalid entries with NaN
    output = tl.math.asin(input)
    output = tl.where(condition, output, float('nan'))
    tl.store(output_ptr + offsets, output, mask=mask)

def asin(input, *, out=None):
    assert input.is_cuda, "Input tensor must be on CUDA"
    assert input.dtype.is_floating_point, "Input tensor must be a floating-point type"
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.dtype == input.dtype, "out tensor must have the same dtype as input"
        assert out.shape == input.shape, "out tensor must have the same shape as input"
        assert out.is_cuda, "out tensor must be on CUDA"
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    asin_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
