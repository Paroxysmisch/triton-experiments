import torch
import triton
import triton.language as tl

@triton.jit
def tanh_kernel(
    input_ptr, output_ptr,
    n_elements,
    input_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input_offset = offsets * input_row_stride
    x = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    output = tl.tanh(x)
    
    output_offset = offsets * output_row_stride
    tl.store(output_ptr + output_offset, output, mask=mask)

def tanh(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    assert input.shape == out.shape, "Input and output shapes must match"
    assert input.device == out.device, "Input and output must be on the same device"
    assert input.dtype == out.dtype, "Input and output dtypes must match"
    
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    tanh_kernel[grid](
        input, out,
        n_elements,
        input.stride(0) if input.ndim == 1 else 1,
        out.stride(0) if out.ndim == 1 else 1,
        BLOCK_SIZE=1024,
    )
    return out

# Example usage:
# input = torch.randn(1024, device='cuda')
# output = tanh(input)
# print(output)
