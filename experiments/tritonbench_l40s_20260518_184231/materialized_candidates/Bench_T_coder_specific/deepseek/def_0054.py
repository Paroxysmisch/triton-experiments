from tritonclient.python.triton import Tensor

def gelu_conv2d(input: Tensor, weight: Tensor, bias: Tensor = None, stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int], str] = 0, dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1, approximate: str = 'none', out: Optional[Tensor] = None) -> Tensor:
    # Define the options
    conv_options = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups
    }

    # Perform the convolution
    output = Tensor.conv2d(input, weight, options=conv_options)

    # Add bias if provided
    if bias is not None:
        output = Tensor.add(output, bias)

    # Apply GELU activation
    if approximate == 'none':
        # Standard GELU
        output = Tensor.gelu(output)
    elif approximate == 'tanh':
        # Approximated GELU
        output = Tensor.gelu_tanh(output)
    else:
        raise ValueError(f"Invalid value for 'approximate': {approximate}")

    return output
