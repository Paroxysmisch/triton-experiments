import triton
import triton.language as tl
from triton.language.math import erf, pow, tanh
import torch

@triton.jit
def gelu_none_kernel(x):
    # Compute the GELU function using the error function approximation
    x_gelu = 0.5 * x * (1 + erf(x * 0.7071067811))
    return x_gelu

@triton.jit
def gelu_tanh_kernel(x):
    # Compute the GELU function using the tanh approximation
    x_gelu = 0.5 * x * (1 + tanh(x * 0.79788456 * (1 + 0.044715 * pow(x, 2))))
    return x_gelu

@triton.jit
def add_gelu_kernel(
    X_ptr, Y_ptr, Z_ptr,
    X_shape, Y_shape, Z_shape,
    alpha, approx,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < min(X_shape[0], Y_shape[0], Z_shape[0])
    x = tl.load(X_ptr + offsets, mask=mask)
    y = alpha * tl.load(Y_ptr + offsets, mask=mask)

    if approx == b'none':
        z = gelu_none_kernel(x + y)
    elif approx == b'tanh':
        z = gelu_tanh_kernel(x + y)
    else:
        raise ValueError(f"Invalid approximate value: {approx.decode()}")

    tl.store(Z_ptr + offsets, z, mask=mask)

class AddGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other, alpha=1, approximate='none'):
        # Get the shapes of the input tensors
        input_shape = input.shape
        other_shape = other.shape
        assert input_shape == other_shape, "Input and other must have the same shape"

        # Allocate output tensor
        output = torch.empty_like(input)
        grid = lambda meta: (triton.cdiv(input.numel(), meta['BLOCK_SIZE']),)
        
        # Launch the kernel
        add_gelu_kernel[grid](input.data_ptr(), other.data_ptr(), output.data_ptr(),
                              input_shape, other_shape, output.shape,
                              alpha, approximate.encode(), 
                              BLOCK_SIZE=256)
        
        ctx.save_for_backward(input, other, alpha, approximate)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, other, alpha, approximate = ctx.saved_tensors
        grad_input = grad_other = None
        
        if ctx.needs_input_grad[0]:
            grad_input = AddGelu.apply(grad_output, other, alpha, approximate)
            
        if ctx.needs_input_grad[1]:
            grad_other = AddGelu.apply(input, grad_output, alpha, approximate)
            
        return grad_input, grad_other, None, None

def add_gelu(input, other, alpha=1, approximate='none', out=None):
    return AddGelu.apply(input, other, alpha, approximate)
