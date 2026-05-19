def mul_sub(input, other_mul, other_sub, alpha=1, out=None):
    # Check if inputs are tensors or numbers
    if not isinstance(other_mul, (Tensor, Number)):
        raise ValueError("other_mul must be a Tensor or Number")
    if not isinstance(other_sub, (Tensor, Number)):
        raise ValueError("other_sub must be a Tensor or Number")

    # Perform element-wise multiplication and subtraction
    if isinstance(other_mul, Tensor):
        result = input.mul(other_mul)
    else:
        result = input.mul(other_mul)
        
    if isinstance(other_sub, Tensor):
        result = result.sub(alpha * other_sub)
    else:
        result = result.sub(alpha * other_sub)

    # Return the result
    return result
