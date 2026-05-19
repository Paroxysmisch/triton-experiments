import triton
import triton.language as tl
from triton.language.math import erf, sqrt, tanh

@triton.jit
def gelu_none_kernel(x):
    # Convert input to float32 for better precision in operations
    x_fp32 = x.to(tl.float32)
    # Compute the GELU function using the error function approximation
    x_gelu = 0.5 * x_fp32 * (1 + erf(x_fp32 * 0.7071067811865475))
    return x_gelu

@triton.jit
def gelu_tanh_kernel(x):
    # Convert input to float32 for better precision in operations
    x_fp32 = x.to(tl.float32)
    # Compute the GELU function using the tanh approximation
    cdf_const = 0.044715
    tanh_arg = sqrt(2 / tl.constexpr(3.14159)) * (x_fp32 + cdf_const * pow(x_fp32, 3))
    x_gelu = 0.5 * x_fp32 * (1 + tanh(tanh_arg))
    return x_gelu

class Gelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, approximate="none"):
        # Choose the kernel based on the approximation method
        if approximate == "none":
            output = gelu_none_kernel(input)
        elif approximate == "tanh":
            output = gelu_tanh_kernel(input)
        else:
            raise ValueError(f"Invalid approximate value: {approximate}")
        
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        # Compute the derivative of GELU
        if ctx.approximate == "none":
            # Using the exact derivative of GELU
            g_x = 0.5 * (1 + erf(input * 0.7071067811865475) + input * (erf(input * 0.7071067811865475) ** 2))
        elif ctx.approximate == "tanh":
            # Using the derivative of the tanh approximation
            cdf_const = 0.044715
            tanh_arg = sqrt(2 / tl.constexpr(3.14159)) * (input + cdf_const * pow(input, 3))
            tanh_val = tanh(tanh_arg)
            sech_val = 1 / (cosh(tanh_arg) ** 2)
            g_x = 0.5 * (1 + tanh_val + input * sech_val * sech_val * (cdf_const * 3 * pow(input, 2) + 1))
        else:
            raise ValueError(f"Invalid approximate value: {ctx.approximate}")
        
        return grad_output * g_x, None

def gelu(input, approximate="none"):
    # Wrapper function for using Gelu class
    return Gelu.apply(input, approximate)
