def abs(input, *, out=None):
    # Get the Triton context
    ctx = get_triton_context()

    # Create the output tensor if not provided
    if out is None:
        out = ctx.empty_like(input)

    # Call the Triton kernel
    call_triton_kernel(out, input, func=triton.ops.abs)

    return out
