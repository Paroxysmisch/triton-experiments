def asin(input, *, out=None):
    # Check if input is a Tensor
    if not isinstance(input, Tensor):
        raise TypeError("input must be a Tensor")
    
    # Check if out is a Tensor
    if out is not None and not isinstance(out, Tensor):
        raise TypeError("out must be a Tensor")
    
    # Call Triton function
    output = triton.ops.asin(input, out=out)
    
    # Check if output is NaN
    if triton.ops.any(triton.ops.isnan(output)):
        raise ValueError("Input values outside the range [-1, 1] are not defined for arcsine")
    
    return output
