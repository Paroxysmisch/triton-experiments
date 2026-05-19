import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    output = 1.0 / (1.0 + tl.exp(-x))
    tl.store(output_ptr + offsets, output, mask=mask)

def sigmoid(input, *, out=None):
    assert input.is_cuda, "Input tensor must be on a CUDA device"
    input = input.contiguous()
    n_elements = input.numel()
    
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_cuda, "Output tensor must be on a CUDA device"
        assert out.shape == input.shape, "Output shape must match input shape"
        out = out.contiguous()
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](input.data_ptr(), out.data_ptr(), n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
