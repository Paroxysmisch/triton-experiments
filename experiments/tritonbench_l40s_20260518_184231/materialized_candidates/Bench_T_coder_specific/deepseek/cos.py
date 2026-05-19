import triton

def cos_wrapper(input, out=None):
    # Compute the cosine of the input tensor
    output = triton.ops.cos(input)

    # If an output tensor is provided, copy the output tensor to it
    if out is not None:
        out[:] = output[:]

    # Return the output tensor
    return output
