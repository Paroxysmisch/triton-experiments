import torch
import triton
import triton.language as tl

@triton.jit
def leaky_relu(x, negative_slope):
    negative_slope = negative_slope.to(x.dtype)
    return tl.where(x >= 0, x, negative_slope * x)

@triton.jit
def leaky_relu_kernel(input_ptr, output_ptr, negative_slope, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    output = leaky_relu(x, negative_slope)
    tl.store(output_ptr + offsets, output, mask=mask)

def leaky_relu(input: torch.Tensor, negative_slope: float = 0.01, inplace: bool = False) -> torch.Tensor:
    if input.is_cuda and triton.is_available():
        if inplace:
            output = input
        else:
            output = torch.empty_like(input)
        n_elements = input.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        leaky_relu_kernel[grid](input.data_ptr(), output.data_ptr(), negative_slope, n_elements, BLOCK_SIZE=1024)
        return output
    else:
        # Fallback to PyTorch's implementation if Triton is not available
        return torch.nn.functional.leaky_relu(input, negative_slope=negative_slope, inplace=inplace)
