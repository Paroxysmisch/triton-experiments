import triton
import triton.language as tl
import torch
import math

@triton.jit
def relu_sqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_values = tl.load(input_ptr + offsets, mask=mask)
    relu_values = tl.maximum(input_values, 0.0)
    sqrt_values = tl.sqrt(relu_values.to(tl.float32))
    tl.store(output_ptr + offsets, sqrt_values, mask=mask)

def relu_sqrt(input, inplace=False, out=None) -> torch.Tensor:
    if inplace:
        if not input.is_contiguous():
            input = input.contiguous()
        if not input.is_floating_point():
            raise RuntimeError("Cannot perform inplace operation on non-floating point tensor")
        output = input
    else:
        if out is not None:
            if not out.is_floating_point():
                raise RuntimeError("out tensor must be a floating point type")
            output = out
        else:
            output_dtype = input.dtype if input.is_floating_point() else torch.float32
            output = torch.empty_like(input, dtype=output_dtype)
    
    n_elements = input.numel()
    if n_elements == 0:
        return output
    
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    
    relu_sqrt_kernel[(grid_size, 1, 1)](input, output, n_elements, BLOCK_SIZE=block_size)
    
    return output
