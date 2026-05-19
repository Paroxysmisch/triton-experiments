import triton
import triton.language as tl
from triton.language.math import erf, pow, tanh
import torch
from typing import Optional, Tuple

# Define the Triton kernels
@triton.jit
def gelu_exact_kernel(x_ptr, out_ptr, n_elements: int):
    pid = tl.program_id(axis=0)
    block_size = tl.block_dim(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_gelu = x * 0.5 * (1 + erf(x * 0.7071067811))
    tl.store(out_ptr + offsets, x_gelu, mask=mask)

@triton.jit
def gelu_tanh_kernel(x_ptr, out_ptr, n_elements: int):
    pid = tl.program_id(axis=0)
    block_size = tl.block_dim(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_gelu = 0.5 * x * (1 + tanh(0.79788456 * x * (1 + 0.044715 * pow(x, 2))))
    tl.store(out_ptr + offsets, x_gelu, mask=mask)

# Define the wrapper function
def gelu_min(input: torch.Tensor, approximate: str = 'none', dim: Optional[int] = None, keepdim: bool = False, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if approximate not in ['none', 'tanh']:
        raise ValueError(f"Invalid approximate value: {approximate}")
    
    if dim is not None:
        output_shape = list(input.shape)
        output_shape[dim] = 1 if keepdim else 0
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device, requires_grad=input.requires_grad)
        
        if approximate == 'none':
            @triton.jit
            def min_reduce_kernel(x_ptr, out_ptr, n_elements: int, stride: int):
                pid = tl.program_id(axis=0)
                block_size = tl.block_dim(axis=0)
                offsets = pid * block_size + tl.arange(0, block_size)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets * stride, mask=mask)
                x_gelu = x * 0.5 * (1 + erf(x * 0.7071067811))
                min_val = tl.min(x_gelu, axis=0)
                tl.store(out_ptr, min_val, mask=True)
            
            n_elements = input.numel()
            grid = (tl.cdiv(n_elements, 1024),)
            min_reduce_kernel[input.stride(dim), :](input.data_ptr(), output.data_ptr(), n_elements, input.stride(dim))
        else:
            @triton.jit
            def min_reduce_kernel(x_ptr, out_ptr, n_elements: int, stride: int):
                pid = tl.program_id(axis=0)
                block_size = tl.block_dim(axis=0)
                offsets = pid * block_size + tl.arange(0, block_size)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets * stride, mask=mask)
                x_gelu = 0.5 * x * (1 + tanh(0.79788456 * x * (1 + 0.044715 * pow(x, 2))))
                min_val = tl.min(x_gelu, axis=0)
                tl.store(out_ptr, min_val, mask=True)
            
            n_elements = input.numel()
            grid = (tl.cdiv(n_elements, 1024),)
            min_reduce_kernel[input.stride(dim), :](input.data_ptr(), output.data_ptr(), n_elements, input.stride(dim))
        
        return output
    else:
        if approximate == 'none':
            @triton.jit
            def reduce_kernel(x_ptr, out_ptr, n_elements: int):
                pid = tl.program_id(axis=0)
                block_size = tl.block_dim(axis=0)
                offsets = pid * block_size + tl.arange(0, block_size)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                x_gelu = x * 0.5 * (1 + erf(x * 0.7071067811))
                min_val = tl.min(x_gelu, axis=0)
                tl.store(out_ptr, min_val, mask=True)
            
            n_elements = input.numel()
            grid = (tl.cdiv(n_elements, 1024),)
            reduce_kernel[input.numel()], :](input.data_ptr(), input.data_ptr(), n_elements)
        else:
            @triton.jit
            def reduce_kernel(x_ptr, out_ptr, n_elements: int):
                pid = tl.program_id(axis=0)
                block_size = tl.block_dim(axis=0)
                offsets = pid * block_size + tl.arange(0, block_size)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                x_gelu = 0.5 * x * (1 + tanh(0.79788456 * x * (1 + 0.044715 * pow(x, 2))))
                min_val = tl.min(x_gelu, axis=0)
                tl.store(out_ptr, min_val, mask=True)
            
            n_elements = input.numel()
            grid = (tl.cdiv(n_elements, 1024),)
            reduce_kernel[input.numel()], :](input.data_ptr(), input.data_ptr(), n_elements)
        
        return input

# Example usage
if __name__ == "__main__":
    input_tensor = torch.randn(1000, device='cuda')
    result = gelu_min(input_tensor, approximate='tanh', dim=0, keepdim=True)
    print(result)
