import torch
import triton
import triton.language as tl

@triton.jit
def softplus_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    w = tl.load(weight_ptr + offsets, mask=mask)
    b = tl.load(bias_ptr + offsets, mask=mask)
    y = tl.dot(x, w) + b
    z = tl.where(y > 20, y, tl.math.log1p(tl.exp(y)))
    tl.store(output_ptr + offsets, z, mask=mask)

def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    assert input.is_contiguous()
    assert weight.is_contiguous()
    assert bias is None or bias.is_contiguous()
    output = torch.empty_like(input)
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    softplus_linear_kernel[grid](input, weight, bias, output, n_elements, BLOCK_SIZE=1024)
    return output
