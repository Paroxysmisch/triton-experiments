import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_masked_select_add_gelu_kernel(
    input_ptr, other_ptr, mask_indices_ptr, output_ptr,
    alpha, num_selected,
    input_stride, other_stride, mask_indices_stride,
    output_stride, approximate_tanh,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs_base = pid * BLOCK_SIZE
    offs = offs_base + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_selected

    # Load mask indices
    indices = tl.load(mask_indices_ptr + offs * mask_indices_stride, mask=mask, other=0)

    # Load input values using indices
    input_vals = tl.load(input_ptr + indices * input_stride, mask=mask, other=0.0)

    # Load other values
    other_vals = tl.load(other_ptr + offs * other_stride, mask=mask, other=0.0)

    # Compute S = input_val + alpha * other_val
    s = input_vals + alpha * other_vals

    # Compute GELU
    if approximate_tanh:
        # Approximate GELU with tanh
        tanh_arg = tl.math.sqrt(2.0 / math.pi) * (s + 0.044715 * s * s * s)
        tanh_val = tl.tanh(tanh_arg)
        y = 0.5 * s * (1.0 + tanh_val)
    else:
        # Exact GELU using erf
        erf_arg = s / tl.math.sqrt(2.0)
        erf_val = tl.erf(erf_arg)
        y = 0.5 * s * (1.0 + erf_val)

    # Store output
    tl.store(output_ptr + offs * output_stride, y, mask=mask)

class FusedMaskedSelectAddGeluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_flat, mask_indices, other_processed, alpha, approximate_tanh, original_other_is_scalar, original_other_shape, original_input_shape):
        ctx.save_for_backward(input_flat, mask_indices, other_processed)
        ctx.alpha = alpha
        ctx.approximate_tanh = approximate_tanh
        ctx.original_other_is_scalar = original_other_is_scalar
        ctx.original_other_shape = original_other_shape
        ctx.original_input_shape = original_input_shape

        num_selected = mask_indices.size(0)
        output = torch.empty((num_selected,), dtype=input_flat.dtype, device=input_flat.device)

        grid = lambda meta: (triton.cdiv(num_selected, meta['BLOCK_SIZE']),)
        fused_masked_select_add_gelu_kernel[grid](
            input_flat, other_processed, mask_indices, output,
            alpha, num_selected,
            input_flat.stride(0), other_processed.stride(0), mask_indices.stride(0),
            output.stride(0), approximate_tanh,
            BLOCK_SIZE=1024
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_flat, mask_indices, other_processed = ctx.saved_tensors
        alpha = ctx.alpha
        approximate_tanh = ctx.approximate_tanh
        original_other_is_scalar = ctx.original_other_is_scalar
        original_other_shape = ctx.original_other_shape
        original_input_shape = ctx.original_input_shape

        num_selected = mask_indices.size(0)
        if num_selected == 0:
            return torch.zeros_like(input_flat).reshape(original_input_shape), None, None, None, None, None, None, None

        # Recompute Z and S for backward pass
        z = input_flat[mask_indices]
        s = z + alpha * other_processed

        # Compute dY/dS
        if approximate_tanh:
            sqrt_2_over_pi = math.sqrt(2 / math.pi)
            k = sqrt_2_over_pi * (s + 0.044715 * s**3)
            tanh_k = torch.tanh(k)
            dyds = 0.5 * (1 + tanh_k) + 0.5 * s * (1 - tanh_k**2) * sqrt_2_over_pi * (1 + 0.134145 * s**2)
        else:
            erf_term = (s / math.sqrt(2.0)).erf()
            pdf = 1.0 / math.sqrt(2 * math.pi) * torch.exp(-0.5 * s**2)
            dyds = 0.5 * (1 + erf_term) + s * pdf

        grad_s = grad_output * dyds

        # Gradient for input
        grad_input_flat = torch.zeros_like(input_flat)
        grad_input_flat.scatter_(0, mask_indices, grad_s)
        grad_input = grad_input_flat.reshape(original_input_shape)

        # Gradient for other_processed
        grad_other_processed = grad_s * alpha

        # Gradient for original other
        if original_other_is_scalar:
            grad_other = grad_other_processed.sum()
        else:
            grad_other = grad_other_processed.sum(dim=tuple(range(len(grad_other_processed.shape) - len(original_other_shape))), keepdim=True)
            grad_other = grad_other.reshape(original_other_shape) if original_other_shape else grad_other.sum()

        return grad_input, None, grad_other, None, None, None, None, None

def fused_masked_select_add_gelu(input, mask, other, *, alpha=1, approximate='none', out=None):
    original_input_shape = input.shape
    input_flat = input.flatten()
    mask_flat = mask.flatten().bool()

    assert input_flat.shape == mask_flat.shape, "input and mask must have the same shape"

    mask_indices = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
    num_selected = mask_indices.size(0)

    if num_selected == 0:
        output = torch.empty((0,), dtype=input.dtype, device=input.device)
        if out is not None:
            out.copy_(output)
        return output

    original_other_is_scalar = False
    original_other_shape = None
    if isinstance(other, torch.Tensor):
        original_other_shape = other.shape
        try:
            other_processed = torch.broadcast_to(other, (num_selected,))
        except RuntimeError:
            raise RuntimeError("other is not broadcastable to the selected elements' shape")
        other_processed = other_processed.to(dtype=input.dtype, device=input.device).contiguous()
    else:
        original_other_is_scalar = True
        original_other_shape = ()
        other_processed = torch.full((num_selected,), other, dtype=input.dtype, device=input.device)

    approximate_tanh = (approximate == 'tanh')

    output = FusedMaskedSelectAddGeluFunction.apply(
        input_flat, mask_indices, other_processed, alpha, approximate_tanh,
        original_other_is_scalar, original_other_shape, original_input_shape
    )

    if out is not None:
        out.copy_(output)
    return output
