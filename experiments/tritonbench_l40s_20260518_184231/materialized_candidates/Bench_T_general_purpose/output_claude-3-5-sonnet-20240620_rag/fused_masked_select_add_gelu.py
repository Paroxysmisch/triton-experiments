import torch
import triton
import triton.language as tl
from triton.language.math import erf, pow, tanh
from typing import Optional

@triton.jit
def fused_masked_select_add_gelu_kernel_none(
    input_ptr, mask_ptr, other_ptr, output_ptr,
    stride_input, stride_mask, stride_other, stride_out,
    n_elements, alpha,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input, mask, and other tensors
    input_offset = offsets * stride_input
    mask_offset = offsets * stride_mask
    other_offset = offsets * stride_other
    
    x = tl.load(input_ptr + input_offset, mask=mask)
    m = tl.load(mask_ptr + mask_offset, mask=mask)
    o = tl.load(other_ptr + other_offset, mask=mask)

    # Apply mask and compute masked add
    x = tl.where(m, x, 0.0)
    x = x + alpha * o

    # Convert to float32 for better precision in GELU computation
    x_fp32 = x.to(tl.float32)
    # Compute GELU using error function
    result = 0.5 * x_fp32 * (1.0 + erf(x_fp32 * 0.7071067811865476))
    
    # Store result
    output_offset = offsets * stride_out
    tl.store(output_ptr + output_offset, result, mask=mask)

@triton.jit
def fused_masked_select_add_gelu_kernel_tanh(
    input_ptr, mask_ptr, other_ptr, output_ptr,
    stride_input, stride_mask, stride_other, stride_out,
    n_elements, alpha,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input, mask, and other tensors
    input_offset = offsets * stride_input
    mask_offset = offsets * stride_mask
    other_offset = offsets * stride_other
    
    x = tl.load(input_ptr + input_offset, mask=mask)
    m = tl.load(mask_ptr + mask_offset, mask=mask)
    o = tl.load(other_ptr + other_offset, mask=mask)

    # Apply mask and compute masked add
    x = tl.where(m, x, 0.0)
    x = x + alpha * o

    # Convert to float32 for better precision in GELU computation
    x_fp32 = x.to(tl.float32)
    # Compute GELU using tanh approximation
    result = 0.5 * x_fp32 * (1.0 + tanh(
        0.7978845608028654 * x_fp32 * (1.0 + 0.044715 * x_fp32 * x_fp32)
    ))
    
    # Store result
    output_offset = offsets * stride_out
    tl.store(output_ptr + output_offset, result, mask=mask)

class FusedMaskedSelectAddGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, mask: torch.Tensor, other: torch.Tensor, 
                alpha: float = 1.0, approximate: str = 'none') -> torch.Tensor:
        # Input validation
        if not mask.dtype == torch.bool:
            raise TypeError("mask must be a boolean tensor")
        if not input.is_contiguous() or not mask.is_contiguous() or not other.is_contiguous():
            raise RuntimeError("All input tensors must be contiguous")
        
        # Create output tensor
        output = torch.empty_like(input)
        
        # Grid and block sizes
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(input.numel(), BLOCK_SIZE),)
        
        # Select kernel based on approximation method
        kernel = (fused_masked_select_add_gelu_kernel_tanh 
                 if approximate == 'tanh' 
                 else fused_masked_select_add_gelu_kernel_none)
        
        # Launch kernel
        kernel[grid](
            input.data_ptr(),
            mask.data_ptr(),
            other.data_ptr(),
            output.data_ptr(),
            input.stride(0),
            mask.stride(0),
            other.stride(0),
            output.stride(0),
            input.numel(),
            alpha,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output

def fused_masked_select_add_gelu(
    input: torch.Tensor,
    mask: torch.Tensor,
    other: torch.Tensor,
    *,
    alpha: float = 1.0,
    approximate: str = 'none',
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Performs a fused masked selection, addition, and GELU activation operation.
    
    Args:
        input (Tensor): Input tensor
        mask (Tensor): Boolean mask tensor
        other (Tensor): Tensor to be added (after scaling by alpha)
        alpha (float, optional): Scaling factor for other tensor. Default: 1.0
        approximate (str, optional): GELU approximation method ('none' or 'tanh'). Default: 'none'
        out (Tensor, optional): Output tensor. Default: None
    
    Returns:
        Tensor: Result of the fused operation
    """
    if approximate not in ['none', 'tanh']:
        raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate}")
    
    if out is not None:
        raise NotImplementedError("out parameter is not supported yet")
    
    return FusedMaskedSelectAddGelu.apply(input, mask, other, alpha, approximate)
