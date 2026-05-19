import torch
import triton
import triton.language as tl
from triton.language.math import erf, tanh

@triton.jit
def add_gelu_none_kernel(
    input_ptr, other_ptr, output_ptr,
    alpha,
    input_row_stride, other_row_stride, output_row_stride,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input = tl.load(input_ptr + offsets * input_row_stride, mask=mask)
    other = tl.load(other_ptr + offsets * other_row_stride, mask=mask)
    
    sum_val = input + alpha * other
    sum_fp32 = sum_val.to(tl.float32)
    
    gelu = 0.5 * sum_fp32 * (1.0 + erf(sum_fp32 * 0.7071067811))
    
    tl.store(output_ptr + offsets * output_row_stride, gelu, mask=mask)

@triton.jit
def add_gelu_tanh_kernel(
    input_ptr, other_ptr, output_ptr,
    alpha,
    input_row_stride, other_row_stride, output_row_stride,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input = tl.load(input_ptr + offsets * input_row_stride, mask=mask)
    other = tl.load(other_ptr + offsets * other_row_stride, mask=mask)
    
    sum_val = input + alpha * other
    sum_fp32 = sum_val.to(tl.float32)
    x = sum_fp32
    
    tanh_arg = 0.79788456 * x * (1.0 + 0.044715 * x * x)
    tanh_val = tl.tanh(tanh_arg)
    gelu = 0.5 * x * (1.0 + tanh_val)
    
    tl.store(output_ptr + offsets * output_row_stride, gelu, mask=mask)

def add_gelu(input, other, alpha=1, approximate='none', out=None) -> torch.Tensor:
    # Ensure other is a tensor and on the correct device/dtype
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    other = other.to(device=input.device, dtype=input.dtype)
    
    # Compute broadcast shape and expand inputs
    broadcast_shape = torch.broadcast_shapes(input.shape, other.shape)
    input_expanded = input.expand(broadcast_shape).contiguous()
    other_expanded = other.expand(broadcast_shape).contiguous()
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(input_expanded)
    else:
        if out.shape != broadcast_shape:
            raise RuntimeError("out shape does not match broadcast shape")
        if not out.is_contiguous():
            raise RuntimeError("out tensor must be contiguous")
    
    n_elements = input_expanded.numel()
    
    # Flatten tensors for kernel processing
    input_flat = input_expanded.view(-1)
    other_flat = other_expanded.view(-1)
    out_flat = out.view(-1)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    BLOCK_SIZE = 1024
    
    if approximate == 'none':
        add_gelu_none_kernel[grid](
            input_flat, other_flat, out_flat,
            alpha,
            1, 1, 1,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif approximate == 'tanh':
        add_gelu_tanh_kernel[grid](
            input_flat, other_flat, out_flat,
            alpha,
            1, 1, 1,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        raise ValueError(f"Invalid approximate method: {approximate}")
    
    return out
