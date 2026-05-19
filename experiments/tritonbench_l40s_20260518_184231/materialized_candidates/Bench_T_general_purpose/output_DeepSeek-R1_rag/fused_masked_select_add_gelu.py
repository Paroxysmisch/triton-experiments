import torch
import triton
import triton.language as tl
from torch.cuda import synchronize

@triton.jit
def masked_select_add_gelu_forward_kernel(
    input_ptr, other_ptr, output_ptr, mask_ptr,
    alpha, input_size, BLOCK_SIZE: tl.constexpr,
    approximate_none: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_size

    # Load mask value
    mask_vals = tl.load(mask_ptr + offsets, mask=mask, other=0)
    mask_filter = mask_vals != 0

    # Load input and other values
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    other_vals = tl.load(other_ptr + offsets, mask=mask, other=0)

    # Compute S = input + alpha * other
    s = input_vals + alpha * other_vals

    # Compute GELU
    if approximate_none:
        # GELU using ERF
        gelu = 0.5 * s * (1 + tl.erf(s * 0.7071067811))
    else:
        # GELU using tanh approximation
        tanh_arg = s * 0.79788456 * (1 + 0.044715 * s * s)
        tanh_val = tl.tanh(tanh_arg)
        gelu = 0.5 * s * (1 + tanh_val)

    # Write to output only where mask is True
    tl.store(output_ptr + offsets, gelu, mask=mask_filter)

class FusedMaskedSelectAddGeluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask, other, alpha, approximate):
        # Ensure other is a tensor
        if not isinstance(other, torch.Tensor):
            other = torch.tensor(other, device=input.device, dtype=input.dtype)
        
        # Expand other to input's shape
        try:
            other_expanded = other.expand_as(input).contiguous()
        except RuntimeError as e:
            raise RuntimeError(f"other tensor shape {other.shape} cannot be broadcast to input shape {input.shape}") from e
        
        # Prepare mask
        mask = mask.bool()
        mask_expanded = mask.expand_as(input).contiguous()
        output = torch.empty_like(input).masked_fill_(~mask, 0)
        
        # Flatten all tensors
        input_flat = input.contiguous().view(-1)
        other_flat = other_expanded.contiguous().view(-1)
        mask_flat = mask_expanded.contiguous().view(-1)
        output_flat = output.contiguous().view(-1)
        input_size = input_flat.numel()
        
        # Kernel configuration
        BLOCK_SIZE = 128
        grid = (triton.cdiv(input_size, BLOCK_SIZE),)
        approximate_none = approximate == 'none'
        
        # Launch kernel
        masked_select_add_gelu_forward_kernel[grid](
            input_flat, other_flat, output_flat, mask_flat,
            alpha, input_size, BLOCK_SIZE=BLOCK_SIZE,
            approximate_none=approximate_none
        )
        
        # Save for backward
        ctx.save_for_backward(mask, input, other_expanded, output)
        ctx.alpha = alpha
        ctx.approximate = approximate
        ctx.other_requires_grad = other.requires_grad
        ctx.other_shape = other.shape
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, input, other_expanded, output = ctx.saved_tensors
        alpha = ctx.alpha
        approximate = ctx.approximate
        other_requires_grad = ctx.other_requires_grad
        other_shape = ctx.other_shape
        
        # Prepare gradients
        grad_input = torch.zeros_like(input)
        grad_other = torch.zeros_like(other_expanded) if other_requires_grad else None
        
        # Compute gradient of GELU
        s = output / (0.5 * (1 + (output / (input + alpha * other_expanded + 1e-6))))  # Recompute S from output
        if approximate == 'none':
            erf_deriv = 2 / (torch.pi ** 0.5) * torch.exp(-0.5 * s ** 2)
            dgelu_ds = 0.5 * (1 + torch.erf(s / 2 ** 0.5)) + 0.5 * s * erf_deriv / 2 ** 0.5
        else:
            tanh_arg = s * 0.79788456 * (1 + 0.044715 * s ** 2)
            tanh_val = torch.tanh(tanh_arg)
            dtanh_darg = 1 - tanh_val ** 2
            darg_ds = 0.79788456 * (1 + 0.044715 * s ** 2) + s * 0.79788456 * 0.08943 * s
            dgelu_ds = 0.5 * (1 + tanh_val) + 0.5 * s * dtanh_darg * darg_ds
        
        dS = grad_output * dgelu_ds
        
        # Compute gradients for input and other
        grad_input.masked_scatter_(mask, dS.masked_select(mask))
        if other_requires_grad:
            grad_other.masked_scatter_(mask, (dS * alpha).masked_select(mask))
            grad_other = grad_other.sum(dim=tuple(range(grad_other.dim() - len(other_shape)))).view(other_shape)
        
        return grad_input, None, grad_other, None, None

def fused_masked_select_add_gelu(input, mask, other, *, alpha=1, approximate='none', out=None):
    if out is not None:
        return FusedMaskedSelectAddGeluFunction.apply(input, mask, other, alpha, approximate).copy_(out)
    return FusedMaskedSelectAddGeluFunction.apply(input, mask, other, alpha, approximate)
