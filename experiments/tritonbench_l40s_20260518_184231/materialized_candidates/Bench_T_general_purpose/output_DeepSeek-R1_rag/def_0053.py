import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def mul_relu_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask)
    other_vals = tl.load(other_ptr + offsets, mask=mask)

    product = input_vals * other_vals
    zero = tl.zeros_like(product)
    relu_product = tl.maximum(product, zero)

    tl.store(output_ptr + offsets, relu_product, mask=mask)

@torch.inference_mode()
def mul_relu(input: Tensor, other: Tensor | float, inplace: bool = False, out: Tensor = None) -> Tensor:
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Determine output tensor
    if inplace:
        output = input
    else:
        if out is not None:
            assert out.is_contiguous(), "Output tensor must be contiguous."
            output = out
        else:
            output = torch.empty_like(input)
    
    # Process 'other' tensor
    if not isinstance(other, Tensor):
        other_tensor = torch.full_like(input, other)
    else:
        other_tensor = other.expand_as(input).contiguous()
        other_tensor = other_tensor.to(device=input.device, dtype=input.dtype)
    
    assert input.shape == other_tensor.shape, "Shapes must be broadcastable to the same shape."
    
    # Flatten tensors for 1D processing
    input_flat = input.view(-1)
    other_flat = other_tensor.view(-1)
    output_flat = output.view(-1)
    
    num_elements = input_flat.numel()
    if num_elements == 0:
        return output  # Handle empty tensor
    
    # Kernel configuration
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    # Kernel meta for execution context
    device = input.device
    device_idx = device.index
    stream = get_cuda_stream(device_idx)
    kernel_meta = dict(device=device, device_type="cuda", stream=stream)
    
    mul_relu_kernel[grid](
        input_flat,
        other_flat,
        output_flat,
        num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
        **kernel_meta,
    )
    
    return output
