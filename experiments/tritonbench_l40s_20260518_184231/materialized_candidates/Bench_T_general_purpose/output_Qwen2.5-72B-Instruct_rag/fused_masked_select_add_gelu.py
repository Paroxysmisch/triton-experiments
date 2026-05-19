import torch
import triton
import triton.language as tl

@triton.jit
def fused_masked_select_add_gelu_kernel(
    input_ptr, mask_ptr, other_ptr, output_ptr, alpha, approximate,
    input_stride, mask_stride, other_stride, output_stride,
    input_n_elements, mask_n_elements, other_n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_offsets = offsets % mask_n_elements
    input_offsets = offsets % input_n_elements
    other_offsets = offsets % other_n_elements

    input_mask = mask_offsets < mask_n_elements
    mask = tl.load(mask_ptr + mask_offsets, mask=input_mask)
    input = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
    other = tl.load(other_ptr + other_offsets, mask=input_mask, other=0.0)

    selected = tl.where(mask, input, 0.0)
    scaled_other = alpha * other
    added = selected + scaled_other

    if approximate == "none":
        added_fp32 = added.to(tl.float32)
        gelu_result = 0.5 * added_fp32 * (1 + tl.math.erf(added_fp32 * 0.7071067811))
    elif approximate == "tanh":
        added_fp32 = added.to(tl.float32)
        gelu_result = 0.5 * added_fp32 * (1 + tl.math.tanh(0.79788456 * added_fp32 * (1 + 0.044715 * tl.math.pow(added_fp32, 2))))
    else:
        raise ValueError(f"Invalid approximate value: {approximate}")

    output_offsets = offsets % output_n_elements
    tl.store(output_ptr + output_offsets, gelu_result, mask=input_mask)

import torch
import triton
import triton.language as tl

class FusedMaskedSelectAddGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask, other, alpha=1, approximate='none', out=None):
        # Save the inputs for backward pass
        ctx.save_for_backward(input, mask, other)
        ctx.alpha = alpha
        ctx.approximate = approximate

        # Determine the output tensor
        if out is None:
            out = torch.empty_like(input)

        # Launch the Triton kernel
        grid = (triton.cdiv(input.numel(), 1024),)
        fused_masked_select_add_gelu_kernel[grid](
            input, mask, other, out, alpha, approximate,
            input.stride(0), mask.stride(0), other.stride(0), out.stride(0),
            input.numel(), mask.numel(), other.numel(), 1024
        )

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, mask, other = ctx.saved_tensors
        alpha = ctx.alpha
        approximate = ctx.approximate

        # Compute the gradient with respect to the input
        grad_input = torch.zeros_like(input)
        grad_other = torch.zeros_like(other)

        # Launch the Triton kernel for the backward pass
        grid = (triton.cdiv(input.numel(), 1024),)
        fused_masked_select_add_gelu_backward_kernel[grid](
            grad_output, input, mask, other, alpha, approximate,
            grad_input, grad_other,
            input.stride(0), mask.stride(0), other.stride(0), grad_input.stride(0), grad_other.stride(0),
            input.numel(), mask.numel(), other.numel(), 1024
        )

        return grad_input, None, grad_other, None, None, None

def fused_masked_select_add_gelu(input, mask, other, *, alpha=1, approximate='none', out=None):
    return FusedMaskedSelectAddGelu.apply(input, mask, other, alpha, approximate, out)

@triton.jit
def fused_masked_select_add_gelu_backward_kernel(
    grad_output_ptr, input_ptr, mask_ptr, other_ptr, alpha, approximate,
    grad_input_ptr, grad_other_ptr,
    input_stride, mask_stride, other_stride, grad_input_stride, grad_other_stride,
    input_n_elements, mask_n_elements, other_n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_offsets = offsets % mask_n_elements
    input_offsets = offsets % input_n_elements
    other_offsets = offsets % other_n_elements

    input_mask = mask_offsets < mask_n_elements
    mask = tl.load(mask_ptr + mask_offsets, mask=input_mask)
    input = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
    other = tl.load(other_ptr + other_offsets, mask=input_mask, other=0.0)
    grad_output = tl.load(grad_output_ptr + offsets, mask=input_mask, other=0.0)

    selected = tl.where(mask, input, 0.0)
    scaled_other = alpha * other
    added = selected + scaled_other

    if approximate == "none":
        added_fp32 = added.to(tl.float32)
        gelu_derivative = 0.5 * (1 + tl.math.erf(added_fp32 * 0.7071067811)) + 0.5 * added_fp32 * (2 / tl.math.sqrt(tl.math.pi)) * tl.math.exp(-0.5 * tl.math.pow(added_fp32, 2))
    elif approximate == "tanh":
        added_fp32 = added.to(tl.float32)
        tanh_term = tl.math.tanh(0.79788456 * added_fp32 * (1 + 0.044715 * tl.math.pow(added_fp32, 2)))
        gelu_derivative = 0.5 * (1 + tanh_term) + 0.5 * added_fp32 * (1 - tl.math.pow(tanh_term, 2)) * 0.79788456 * (1 + 0.044715 * 3 * tl.math.pow(added_fp32, 2))
    else:
        raise ValueError(f"Invalid approximate value: {approximate}")

    grad_input = grad_output * gelu_derivative
    grad_other = grad_output * gelu_derivative * alpha

    grad_input_offsets = offsets % input_n_elements
    grad_other_offsets = offsets % other_n_elements
    tl.store(grad_input_ptr + grad_input_offsets, grad_input, mask=input_mask)
    tl.store(grad_other_ptr + grad_other_offsets, grad_other, mask=input_mask)

# Test case
input = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
mask = torch.tensor([True, False, True, False])
other = torch.tensor([0.5, 0.5, 0.5, 0.5])
alpha = 2.0
approximate = 'none'

output = fused_masked_select_add_gelu(input, mask, other, alpha=alpha, approximate=approximate)
print("Output:", output)

# Compute the gradient
output.sum().backward()
print("Gradient with respect to input:", input.grad)
print("Gradient with respect to other:", other.grad)
