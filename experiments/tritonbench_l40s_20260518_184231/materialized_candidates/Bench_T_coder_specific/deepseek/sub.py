def sub(input, other, *, alpha=1, out=None):
    # Convert inputs to Triton tensors
    input = triton.as_tensor(input)
    other = triton.as_tensor(other)
    alpha = triton.as_tensor(alpha)
    if out is not None:
        out = triton.as_tensor(out)

    # Perform subtraction operation
    triton.ops.sub(input, other, alpha, out)

    # Return output tensor
    return out if out is not None else input
