import triton

def pow(input, exponent, out=None):
    # check if input and exponent are of correct types
    if not isinstance(input, triton.Tensor):
        raise TypeError("input must be a Tensor")
    if not isinstance(exponent, (float, triton.Tensor)):
        raise TypeError("exponent must be a float or a Tensor")

    # check if out is a Tensor if provided
    if out is not None and not isinstance(out, triton.Tensor):
        raise TypeError("out must be a Tensor if provided")

    # call the triton function
    return triton.pow(input, exponent, out)
