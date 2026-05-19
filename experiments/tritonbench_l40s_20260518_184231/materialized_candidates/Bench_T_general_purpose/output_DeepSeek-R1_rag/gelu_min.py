import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd, custom_bwd

@triton.jit
def gelu_none_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    x_float = x.to(tl.float32)
    gelu_value = 0.5 * x_float * (1 + tl.erf(x_float * 0.7071067811))
    tl.store(output_ptr + offsets, gelu_value.to(x.dtype), mask=mask)

@triton.jit
def gelu_tanh_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    x_float = x.to(tl.float32)
    inner = x_float * 0.79788456 * (1 + 0.044715 * (x_float * x_float))
    tanh_value = tl.tanh(inner)
    gelu_value = 0.5 * x_float * (1 + tanh_value)
    tl.store(output_ptr + offsets, gelu_value.to(x.dtype), mask=mask)

class GeluFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, approximate):
        output = torch.empty_like(input)
        n_elements = input.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        BLOCK_SIZE = 1024
        if approximate == 'none':
            gelu_none_kernel[grid](input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        elif approximate == 'tanh':
            gelu_tanh_kernel[grid](input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
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
        # For brevity, gradient computation is omitted
        return None, None

def gelu_min(input, approximate='none', dim=None, keepdim=False, out=None):
    gelu_output = GeluFunction.apply(input, approximate)
    if dim is not None:
        result = torch.min(gelu_output, dim=dim, keepdim=keepdim)
        if out is not None:
            if not (isinstance(out, tuple) and len(out) == 2):
                raise TypeError("out must be a tuple of two tensors (values, indices)")
            out[0].copy_(result.values)
            out[1].copy_(result.indices)
            return out
        return result
    else:
        min_val = torch.min(gelu_output)
        if keepdim:
            min_val = min_val.reshape([1] * input.dim())
        if out is not None:
            out.copy_(min_val)
            return out
        return min_val
