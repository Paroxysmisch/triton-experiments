import torch
import triton
import triton.language as tl

@triton.jit
def sqrt_exp_kernel(input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    sqrt_x = tl.sqrt(x)
    exp_sqrt_x = tl.exp(sqrt_x)
    tl.store(output_ptr + offsets, exp_sqrt_x, mask=mask)

def sqrt_exp(input, out=None) -> torch.Tensor:
    input = input.contiguous()
    if not input.is_cuda:
        raise ValueError("Input tensor must be on CUDA")
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.size() != input.size():
            raise ValueError("Output tensor must have the same shape as input")
        if out.dtype != input.dtype:
            raise ValueError("Output tensor must have the same dtype as input")
        if not out.is_contiguous():
            raise ValueError("Output tensor must be contiguous")
        if not out.is_cuda:
            raise ValueError("Output tensor must be on CUDA")
    num_elements = input.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    sqrt_exp_kernel[grid](input.data_ptr(), out.data_ptr(), num_elements, BLOCK_SIZE=1024)
    return out
