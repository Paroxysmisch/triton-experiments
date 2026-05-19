import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd

@triton.jit
def gelu_none_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_float = x.to(tl.float32)
    # Compute GELU using the error function
    gelu_val = x_float * 0.5 * (1 + tl.erf(x_float * 0.7071067811865475))
    gelu_val = gelu_val.to(x.dtype)  # Cast back to original dtype
    tl.store(output_ptr + offsets, gelu_val, mask=mask)

@triton.jit
def gelu_tanh_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_float = x.to(tl.float32)
    # Compute GELU using tanh approximation
    sqrt_2_over_pi = 0.7978845608
    coeff = 0.044715
    inner = x_float * sqrt_2_over_pi * (1 + coeff * x_float * x_float)
    tanh_val = tl.tanh(inner)
    gelu_val = 0.5 * x_float * (1 + tanh_val)
    gelu_val = gelu_val.to(x.dtype)  # Cast back to original dtype
    tl.store(output_ptr + offsets, gelu_val, mask=mask)

class GeluFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, approximate):
        ctx.save_for_backward(x)
        ctx.approximate = approximate
        output = torch.empty_like(x)
        n_elements = x.numel()
        if n_elements == 0:
            return output
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        if approximate == 'none':
            gelu_none_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        elif approximate == 'tanh':
            gelu_tanh_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        else:
            raise ValueError(f"Invalid approximate method: {approximate}")
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        approximate = ctx.approximate
        # For simplicity, using PyTorch's autograd for backward pass
        with torch.enable_grad():
            x_requires_grad = x.detach().requires_grad_(True)
            gelu_x = GeluFunction.apply(x_requires_grad, approximate)
            grad_input = torch.autograd.grad(gelu_x, x_requires_grad, grad_output)[0]
        return grad_input, None

def gelu(input, approximate='none'):
    return GeluFunction.apply(input, approximate)
