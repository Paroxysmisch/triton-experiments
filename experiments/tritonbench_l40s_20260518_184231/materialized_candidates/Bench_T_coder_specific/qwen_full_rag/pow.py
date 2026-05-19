import triton
import triton.language as tl
import torch

@triton.jit
def pow_kernel(output_ptr, input_ptr, exponent,
               input_row_stride, output_row_stride, n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(input_ptr + offsets * input_row_stride, mask=mask)
    # Computation
    y = tl.libdevice.pow(x, exponent)
    # Write-back output
    tl.store(output_ptr + offsets * output_row_stride, y, mask=mask)

def pow(input, exponent, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape
    assert out.is_contiguous()
    _, n_elements = out.view(-1).shape
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    pow_kernel[grid](out, input, exponent,
                     input.stride(0), out.stride(0), n_elements,
                     BLOCK_SIZE=1024)
    return out
