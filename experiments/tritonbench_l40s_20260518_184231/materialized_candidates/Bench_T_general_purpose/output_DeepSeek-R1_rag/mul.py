import triton
import triton.language as tl
import torch
import math

@triton.jit
def mul_tensor_kernel(input_ptr, other_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(input_ptr + offsets, mask=mask)
    b = tl.load(other_ptr + offsets, mask=mask)
    c = a * b
    tl.store(output_ptr + offsets, c, mask=mask)

@triton.jit
def mul_scalar_kernel(input_ptr, other_scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(input_ptr + offsets, mask=mask)
    c = a * other_scalar
    tl.store(output_ptr + offsets, c, mask=mask)

def mul(input, other, *, out=None):
    if isinstance(other, torch.Tensor):
        broadcasted_shape = torch.broadcast_shapes(input.shape, other.shape)
    else:
        broadcasted_shape = input.shape
    
    if out is None:
        dtype = torch.result_type(input, other)
        out = torch.empty(broadcasted_shape, dtype=dtype, device=input.device)
    else:
        assert out.shape == broadcasted_shape, "Output shape must match broadcasted shape"
    
    input_expanded = input.broadcast_to(broadcasted_shape)
    if not input_expanded.is_contiguous():
        input_expanded = input_expanded.contiguous()
    
    if isinstance(other, torch.Tensor):
        other_expanded = other.broadcast_to(broadcasted_shape)
        if not other_expanded.is_contiguous():
            other_expanded = other_expanded.contiguous()
        other_flat = other_expanded.view(-1)
    else:
        other_flat = other
    
    input_flat = input_expanded.view(-1)
    out_flat = out.view(-1)
    n_elements = out_flat.numel()
    
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    
    if isinstance(other, torch.Tensor):
        mul_tensor_kernel[(grid_size, 1, 1)](input_flat, other_flat, out_flat, n_elements, block_size)
    else:
        mul_scalar_kernel[(grid_size, 1, 1)](input_flat, other_flat, out_flat, n_elements, block_size)
    
    return out
