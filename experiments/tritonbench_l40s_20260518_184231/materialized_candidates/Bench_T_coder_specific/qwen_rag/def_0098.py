import triton
import triton.language as tl
from triton.language.math import erf, pow, tanh

@triton.jit
def gelu_none_and_mul_kernel(x, y):
    # Convert input to float32 for better precision in operations
    x_fp32 = x.to(tl.float32)
    # Compute the GELU function using the error function approximation
    x_gelu = 0.5 * x_fp32 * (1 + erf(x_fp32 * 0.7071067811))
    # Multiply the result by y and return
    return x_gelu * y

@triton.jit
def gelu_tanh_and_mul_kernel(x, y):
    # Convert input to float32 for better precision in operations
    x_fp32 = x.to(tl.float32)
    # Compute the GELU function using the tanh approximation
    x_gelu = (
        0.5
        * x_fp32
        * (
            1
            + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * pow(x_fp32.to(tl.float32), 2)))
        )
    )
    # Multiply the result by y and return
    return x_gelu * y

class GeluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other, alpha=1, approximate="none", out=None):
        # Apply subtraction
        sub_result = input - alpha * other
        # Select the appropriate kernel based on the approximation method
        if approximate == "none":
            gelu_result = gelu_none_and_mul_kernel(sub_result, 1)
        elif approximate == "tanh":
            gelu_result = gelu_tanh_and_mul_kernel(sub_result, 1)
        else:
            raise ValueError(f"Invalid approximate value: {approximate}")
        
        # Store the result in the output tensor if provided, otherwise create a new one
        if out is None:
            out = input.new_tensor(gelu_result.shape, dtype=input.dtype)
        out.copy_(gelu_result)
        
        return out

# Example usage
# input_tensor = torch.randn(10, device='cuda')
# other_tensor = torch.randn(10, device='cuda')
# result = sub_gelu(input_tensor, other_tensor, alpha=0.5, approximate='tanh')
