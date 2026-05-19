def reciprocal(input, *, out=None):
    # Check if input is integral and promote to default scalar type if True
    if input.dtype.is_integral:
        input = input.to(torch.get_default_dtype())

    # Call the Triton function
    return torch.reciprocal(input, out=out)
