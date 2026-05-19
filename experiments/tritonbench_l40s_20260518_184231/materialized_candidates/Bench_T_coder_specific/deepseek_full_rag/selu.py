import triton
import triton.language as tl

@triton.jit
def selu(input, inplace=False):
    # Constants for the SELU function
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    # Apply the SELU function element-wise to the input tensor
    if inplace:
        output = input
        input.mul_(scale).where_(input > 0, input.mul(alpha * tl.exp(input) - 1))
    else:
        output = scale * input.where(input > 0, alpha * tl.exp(input) - 1)
    return output

def selu(input, inplace=False):
    # Wrapper function for the Triton kernel
    return selu(input, inplace)
