import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd, custom_bwd

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
    gelu = x_float * 0.5 * (1.0 + tl.math.erf(x_float * 0.7071067811865475))
    output = gelu.to(x.dtype)
    tl.store(output_ptr + offsets, output, mask=mask)

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
    inner = x_float + 0.044715 * x_float * x_float * x_float
    inner *= 0.7978845608
    tanh_inner = tl.math.tanh(inner)
    gelu = 0.5 * x_float * (1.0 + tanh_inner)
    output = gelu.to(x.dtype)
    tl.store(output_ptr + offsets, output, mask=mask)

class GeluFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, approximate):
        output = torch.empty_like(input)
        n_elements = input.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        x_flat = input.view(-1)
        output_flat = output.view(-1)
        if approximate == 'none':
            gelu_none_kernel[grid](x_flat, output_flat, n_elements, BLOCK_SIZE=1024)
        elif approximate == 'tanh':
            gelu_tanh_kernel[grid](x_flat, output_flat, n_elements, BLOCK_SIZE=1024)
        else:
            raise ValueError(f"Invalid approximate method: {approximate}")
        ctx.save_for_backward(input)
        ctx.approximate = approximate
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        approximate = ctx.approximate
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.empty_like(input)
            n_elements = input.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            # Placeholder for backward pass implementation
            # Actual backward kernel required for correct gradients
        return grad_input, None

def min_gelu(input, dim=None, keepdim=False, approximate='none', out=None):
    gelu_output = GeluFunction.apply(input, approximate)
    if dim is not None:
        result = torch.min(gelu_output, dim=dim, keepdim=keepdim)
        return result
    else:
        min_val = torch.min(gelu_output)
        if out is not None:
            out.copy_(min_val)
            return out
        return min_val
