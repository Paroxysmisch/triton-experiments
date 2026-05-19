import torch
import triton
import triton.language as tl
import math
from torch import Tensor

@triton.jit
def std_kernel(
    input_ptr,
    sum_x_ptr,
    sum_x2_ptr,
    n_elements,
    correction,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x_float = x.to(tl.float32)

    sum_x = tl.sum(x_float, 0)
    sum_x2 = tl.sum(x_float * x_float, 0)

    tl.atomic_add(sum_x_ptr, sum_x)
    tl.atomic_add(sum_x2_ptr, sum_x2)

def triton_std(input_1d: Tensor, correction: int = 1) -> Tensor:
    if input_1d.numel() == 0:
        return torch.tensor(float('nan'), device=input_1d.device)
    
    sum_x = torch.zeros(1, dtype=torch.float32, device=input_1d.device)
    sum_x2 = torch.zeros(1, dtype=torch.float32, device=input_1d.device)
    n_elements = input_1d.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    std_kernel[grid](
        input_1d,
        sum_x,
        sum_x2,
        n_elements,
        correction,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    sum_x_val = sum_x.item()
    sum_x2_val = sum_x2.item()
    
    denominator = max(0, n_elements - correction)
    if denominator <= 0:
        return torch.full_like(input_1d[:1], float('inf'), device=input_1d.device)
    
    variance = (sum_x2_val - (sum_x_val ** 2) / n_elements) / denominator
    std_value = math.sqrt(variance) if variance >= 0 else 0.0
    
    return torch.tensor(std_value, dtype=input_1d.dtype, device=input_1d.device)

def sum_std(input: Tensor, dim=None, keepdim=False, dtype=None, correction=1, out=None) -> Tensor:
    if dtype is not None:
        input = input.to(dtype)
    
    sum_result = torch.sum(input, dim=dim, keepdim=keepdim)
    
    if sum_result.numel() == 0:
        std_result = torch.tensor(float('nan'), device=input.device)
    else:
        sum_flat = sum_result.flatten()
        std_result = triton_std(sum_flat, correction=correction)
    
    if keepdim:
        output_shape = sum_result.shape
        std_result = std_result.expand(output_shape)
    
    if out is not None:
        out.copy_(std_result)
        return out
    return std_result
