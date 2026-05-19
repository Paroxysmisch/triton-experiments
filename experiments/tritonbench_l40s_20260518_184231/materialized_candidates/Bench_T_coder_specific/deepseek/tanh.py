def tanh(input, *, out=None):
    # Get the input tensor's shape
    input_shape = input.shape

    # Initialize the output tensor
    if out is None:
        out = input.new_empty(input_shape)
    else:
        assert out.shape == input_shape, "Output tensor must have the same shape as the input tensor"

    # Call the Triton kernel
    triton.ops.tanh(input, out)

    return out
