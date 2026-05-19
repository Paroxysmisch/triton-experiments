def min_gelu(input, dim=None, keepdim=False, approximate='none', out=None):
    # Compute GELU activation
    if approximate == 'none':
        gelu = 0.5 * input * (1 + torch.erf(input / math.sqrt(2)))
    elif approximate == 'tanh':
        gelu = 0.5 * input * (1 + torch.tanh(math.sqrt(2 / math.pi) * (input + 0.044715 * input ** 3)))
    else:
        raise ValueError("Invalid value for 'approximate'")

    # Compute minimum along specified dimension
    if dim is not None:
        return torch.min(gelu, dim=dim, keepdim=keepdim)
    else:
        return torch.min(gelu)
