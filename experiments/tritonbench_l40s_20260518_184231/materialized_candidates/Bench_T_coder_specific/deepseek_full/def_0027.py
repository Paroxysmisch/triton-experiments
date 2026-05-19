import torch
import triton
import triton.language as tl

@triton.jit
def sqrt_tanh(input, out=None):
    # Compute the square root of each element in the input tensor
    input_sqrt = tl.sqrt(input)
    # Apply the hyperbolic tangent function to the square-rooted values
    return tl.tanh(input_sqrt)

# Example usage
input = torch.tensor([0.0, 1.0, -1.0], device='cuda')
output = sqrt_tanh(input)
print(output)
