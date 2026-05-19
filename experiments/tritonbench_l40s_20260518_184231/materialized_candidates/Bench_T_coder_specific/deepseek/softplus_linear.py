import triton
import triton.language as tl

@triton.jit
def softplus_linear(x, weight, bias=None, beta=1, threshold=20):
    # Define the linear transformation
    linear = tl.dot(x, weight)
    if bias is not None:
        linear += bias
    
    # Define the Softplus activation function
    softplus = lambda x: (1/beta) * tl.log(1 + tl.exp(beta * x))

    # Apply the Softplus function to the linear transformation
    out = softplus(linear)

    # Apply a linear function for values above the threshold
    out = tl.where(linear > threshold, linear, out)

    return out
