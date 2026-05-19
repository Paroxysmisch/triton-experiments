import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd, custom_bwd

@triton.jit
def sub_gelu_none_kernel(output_ptr, input_ptr, adjusted_other_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_val = tl.load(input_ptr + offsets, mask=mask)
    adjusted_other_val = tl.load(adjusted_other_ptr + offsets, mask=mask)

    x = input_val - adjusted_other_val
    x = x.to(tl.float32)
    gelu_x = 0.5 * x * (1 + tl.erf(x * 0.7071067811865475))
    tl.store(output_ptr + offsets, gelu_x.to(input_val.dtype), mask=mask)

@triton.jit
def sub_gelu_tanh_kernel(output_ptr, input_ptr, adjusted_other_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_val = tl.load(input_ptr + offsets, mask=mask)
    adjusted_other_val = tl.load(adjusted_other_ptr + offsets, mask=mask)

    x = input_val - adjusted_other_val
    x = x.to(tl.float32)
    x_cubed = x * x * x
    inner = 0.79788456 * (x + 0.044715 * x_cubed)
    tanh_inner = tl.tanh(inner)
    gelu_x = 0.5 * x * (1 + tanh_inner)
    tl.store(output_ptr + offsets, gelu_x.to(input_val.dtype), mask=mask)

def sub_gelu(input, other, alpha=1, approximate='none', out=None):
    assert approximate in ['none', 'tanh'], f"approximate must be 'none' or 'tanh', got {approximate}"
    
    if torch.is_tensor(other):
        adjusted_other = alpha * other
    else:
        adjusted_other = torch.tensor(alpha * other, dtype=input.dtype, device=input.device)
    
    adjusted_other_expanded = adjusted_other.expand_as(input).contiguous()
    
    if out is None:
        output = torch.empty_like(input)
    else:
        output = out
        assert output.is_contiguous(), "Output tensor must be contiguous"
    
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    BLOCK_SIZE = 1024
    
    if approximate == 'none':
        sub_gelu_none_kernel[grid](output, input, adjusted_other_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    else:
        sub_gelu_tanh_kernel[grid](output, input, adjusted_other_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output
