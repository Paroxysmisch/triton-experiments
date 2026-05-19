def add_gelu(input, other, alpha=1, approximate='none', out=None):
    # scale other by alpha
    other = other * alpha

    # add other to input
    result = input + other

    # apply GELU activation function
    if approximate == 'none':
        # exact GELU function
        gelu_result = result * triton.Phi(result)
    elif approximate == 'tanh':
        # approximate GELU function
        gelu_result = 0.5 * result * (1 + triton.Tanh(math.sqrt(2 / math.pi) * (result + 0.044715 * result**3)))
    else:
        raise ValueError("Invalid value for 'approximate'. Must be 'none' or 'tanh'.")

    return gelu_result
