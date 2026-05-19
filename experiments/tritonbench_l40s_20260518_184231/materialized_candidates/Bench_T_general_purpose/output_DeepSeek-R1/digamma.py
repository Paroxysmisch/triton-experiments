import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def digamma_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    zero = tl.zeros(x.shape, x.dtype)
    is_zero = x == zero
    is_negative = x < zero
    x_ge6 = x >= 6.0
    
    # Compute for x >= 6
    log_x = tl.math.log(x)
    term1 = 1.0 / (2.0 * x)
    term2 = 1.0 / (12.0 * x * x)
    approx_ge6 = log_x - term1 - term2
    
    # Compute for x < 6
    six_minus_x = 6.0 - x
    n_float = tl.math.ceil(six_minus_x)
    n_float = tl.where(x_ge6, zero, n_float)
    n_int = n_float.to(tl.int32)
    
    # Calculate sum of reciprocals
    sum_reciprocal = tl.zeros(x.shape, x.dtype)
    for k in range(0, 6):  # Maximum n_int is 6 when x approaches 0
        current_k = tl.full(x.shape, k, tl.int32)
        active = (current_k < n_int) & (~x_ge6)
        denominator = x + current_k.to(x.dtype)
        sum_reciprocal += tl.where(active, 1.0 / denominator, 0.0)
    
    x_new = x + n_int.to(x.dtype)
    log_x_new = tl.math.log(x_new)
    term1_new = 1.0 / (2.0 * x_new)
    term2_new = 1.0 / (12.0 * x_new * x_new)
    approx_x_new = log_x_new - term1_new - term2_new
    approx_lt6 = approx_x_new - sum_reciprocal
    
    # Combine results
    result = tl.where(x_ge6, approx_ge6, approx_lt6)
    result = tl.where(is_zero, -float('inf'), result)
    result = tl.where(is_negative, float('nan'), result)
    
    tl.store(output_ptr + offsets, result, mask=mask)

def digamma(input: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if not input.is_cuda:
        raise NotImplementedError("digamma Triton kernel requires CUDA tensor")
    if input.dtype != torch.float32:
        raise ValueError("Input tensor must be of type float32")
    
    if out is None:
        out = torch.empty_like(input)
    else:
        if not out.is_cuda:
            raise TypeError("out must be a CUDA tensor")
        if out.dtype != torch.float32:
            raise ValueError("Output tensor must be of type float32")
        if out.shape != input.shape:
            raise RuntimeError("out shape must match input shape")
    
    if not input.is_contiguous():
        input = input.contiguous()
    if not out.is_contiguous():
        out = out.contiguous()
    
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    digamma_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    
    return out
