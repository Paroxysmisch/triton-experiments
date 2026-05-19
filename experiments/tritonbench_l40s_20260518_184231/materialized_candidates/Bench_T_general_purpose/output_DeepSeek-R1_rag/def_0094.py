import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd, custom_bwd

@triton.jit
def apply_dropout(input, drop_p, seed, offset):
    random = tl.rand(seed, offset)
    return tl.where(random < drop_p, 0.0, input / (1 - drop_p))

@triton.jit
def apply_dropout_grad(output_grad, drop_p, seed, offset):
    random = tl.rand(seed, offset)
    return tl.where(random < drop_p, 0.0, output_grad / (1 - drop_p))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    ],
    key=['size'],
)
@triton.jit
def dropout_forward_kernel(
    input_pointer, output_pointer, size,
    drop_p, seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    input = tl.load(input_pointer + offset, mask=mask)
    output = apply_dropout(input, drop_p, seed, offset)
    tl.store(output_pointer + offset, output, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    ],
    key=['size'],
)
@triton.jit
def dropout_backward_kernel(
    output_grad_pointer, input_grad_pointer, size,
    drop_p, seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    output_grad = tl.load(output_grad_pointer + offset, mask=mask)
    input_grad = apply_dropout_grad(output_grad, drop_p, seed, offset)
    tl.store(input_grad_pointer + offset, input_grad, mask=mask)

class DropoutSigmoidLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias, p, training, inplace):
        # Linear transformation
        x_linear = F.linear(input, weight, bias)
        # Sigmoid activation
        if inplace:
            x_sigmoid = torch.sigmoid_(x_linear)
        else:
            x_sigmoid = torch.sigmoid(x_linear)
        dropout_applied = False
        seed = None
        if training and p > 0:
            dropout_applied = True
            original_shape = x_sigmoid.shape
            # Ensure tensor is contiguous
            if not x_sigmoid.is_contiguous():
                x_sigmoid = x_sigmoid.contiguous()
            x_flat = x_sigmoid.view(-1)
            size = x_flat.numel()
            # Generate seed
            seed = torch.randint(0, 0x7FFFFFFF, (1,), device=input.device).item()
            # Prepare output
            if inplace:
                output_flat = x_flat
            else:
                output_flat = torch.empty_like(x_flat)
            # Launch kernel
            if size > 0:
                grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
                dropout_forward_kernel[grid](
                    x_flat, output_flat, size, p, seed, BLOCK_SIZE=1024
                )
            x_dropout = output_flat.view(original_shape)
        else:
            x_dropout = x_sigmoid
        # Save for backward
        ctx.save_for_backward(input, weight, bias, x_linear, x_sigmoid)
        ctx.p = p
        ctx.training = training
        ctx.dropout_applied = dropout_applied
        ctx.seed = seed
        ctx.inplace = inplace if dropout_applied else False
        return x_dropout

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias, x_linear, x_sigmoid = ctx.saved_tensors
        p = ctx.p
        training = ctx.training
        dropout_applied = ctx.dropout_applied
        seed = ctx.seed
        inplace = ctx.inplace

        grad_input = grad_weight = grad_bias = None
        # Dropout backward
        if dropout_applied:
            original_shape = grad_output.shape
            if not grad_output.is_contiguous():
                grad_output = grad_output.contiguous()
            grad_flat = grad_output.view(-1)
            size = grad_flat.numel()
            if inplace:
                input_grad_flat = grad_flat
            else:
                input_grad_flat = torch.empty_like(grad_flat)
            # Launch kernel
            if size > 0:
                grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
                dropout_backward_kernel[grid](
                    grad_flat, input_grad_flat, size, p, seed, BLOCK_SIZE=1024
                )
            grad_dropout = input_grad_flat.view(original_shape)
        else:
            grad_dropout = grad_output

        # Sigmoid backward: grad_dropout * sigmoid_deriv
        sigmoid_deriv = x_sigmoid * (1 - x_sigmoid)
        grad_sigmoid = grad_dropout * sigmoid_deriv

        # Linear backward
        if ctx.needs_input_grad[0]:
            grad_input = F.linear(grad_sigmoid, weight)
        if ctx.needs_input_grad[1] or (bias is not None and ctx.needs_input_grad[2]):
            # Flatten batch dimensions for weight and bias gradients
            grad_sigmoid_2d = grad_sigmoid.reshape(-1, grad_sigmoid.size(-1))
            input_2d = input.reshape(-1, input.size(-1))
            if ctx.needs_input_grad[1]:
                grad_weight = grad_sigmoid_2d.t().mm(input_2d)
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_sigmoid_2d.sum(0)
        return grad_input, grad_weight, grad_bias, None, None, None

def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False) -> torch.Tensor:
    return DropoutSigmoidLinearFunction.apply(input, weight, bias, p, training, inplace)
