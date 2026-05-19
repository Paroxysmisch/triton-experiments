import torch
import triton
import triton.language as tl

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def combined_activation(input, weight1, weight2, bias):
    # Apply linear transformation and activation
    intermediate = tanh(tl.dot(input, weight1))
    output = intermediate * weight2 + bias
    return output

def call_combined_activation(input, weight1, weight2, bias):
    # Call the combined activation function
    return combined_activation(input, weight1, weight2, bias)

# Example usage
batch_dims = torch.randn(2, 3, 4)
weight1 = torch.randn(4, 5)
weight2 = torch.randn(5)
bias = torch.randn(5)
output = call_combined_activation(batch_dims, weight1, weight2, bias)
