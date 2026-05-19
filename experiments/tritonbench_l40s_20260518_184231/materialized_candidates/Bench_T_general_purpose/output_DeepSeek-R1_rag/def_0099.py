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
    # Compute GELU using the error function approximation
    gelu = 0.5 * x_float * (1 + tl.erf(x_float * 0.7071067811865475))
    tl.store(output_ptr + offsets, gelu.to(x.dtype), mask=mask)

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
    # Constants for tanh approximation
    alpha = 0.79788456  # sqrt(2/pi)
    beta = 0.044715
    inner = x_float * alpha * (1 + beta * x_float * x_float)
    tanh_inner = tl.tanh(inner)
    gelu = 0.5 * x_float * (1 + tanh_inner)
    tl.store(output_ptr + offsets, gelu.to(x.dtype), mask=mask)

class GeluActivation(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, approximate):
        output = torch.empty_like(input)
        n_elements = output.numel()
        if n_elements == 0:
            return output
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        if approximate == 'none':
            gelu_none_kernel[grid](input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        elif approximate == 'tanh':
            gelu_tanh_kernel[grid](input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        else:
            raise ValueError(f"Invalid approximate method: {approximate}")
        ctx.save_for_backward(input, output)
        ctx.approximate = approximate
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        approximate = ctx.approximate
        if approximate == 'none':
            # Compute derivative using error function approximation
            sqrt_2_over_pi = 0.7978845608  # sqrt(2/pi)
            cdf = 0.5 * (1 + torch.erf(input * 0.7071067811865475))
            pdf = torch.exp(-0.5 * input ** 2) * sqrt_2_over_pi
            dgelu = cdf + input * pdf * 0.5
        elif approximate == 'tanh':
            # Compute derivative using tanh approximation
            alpha = 0.79788456
            beta = 0.044715
            x_sq = input ** 2
            inner = alpha * (input + beta * input ** 3)
            tanh_inner = torch.tanh(inner)
            d_inner = alpha * (1 + 3 * beta * x_sq)
            d_tanh = 1 - tanh_inner ** 2
            dgelu = 0.5 * (1 + tanh_inner) + 0.5 * input * d_tanh * d_inner
        else:
            raise ValueError(f"Invalid approximate method: {approximate}")
        grad_input = grad_output * dgelu
        return grad_input, None

def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None) -> torch.Tensor:
    gelu_output = GeluActivation.apply(input, approximate)
    if dim is None:
        dim = tuple(range(gelu_output.dim()))
    result = torch.std(gelu_output, dim=dim, keepdim=keepdim, correction=correction, out=out)
    return result
