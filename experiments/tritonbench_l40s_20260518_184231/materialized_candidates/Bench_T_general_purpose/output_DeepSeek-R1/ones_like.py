import torch
import triton
import triton.language as tl

@triton.jit
def ones_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 1, mask=mask)

def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> torch.Tensor:
    # Determine the output properties based on input and arguments
    output_dtype = dtype if dtype is not None else input.dtype
    output_device = device if device is not None else input.device
    output_layout = layout if layout is not None else input.layout
    
    # Create an empty tensor with the same properties as input
    output = torch.empty_like(input, dtype=output_dtype, layout=output_layout, device=output_device, memory_format=memory_format)
    
    # Fill with ones using Triton kernel if on CUDA
    if output.is_cuda:
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        ones_kernel[grid](output, n_elements, BLOCK_SIZE=1024)
    else:
        # Fallback to PyTorch's ones_like for CPU tensors
        output = torch.ones_like(input, dtype=output_dtype, layout=output_layout, device=output_device, memory_format=memory_format)
    
    output.requires_grad = requires_grad
    return output
