import triton
import triton.language as tl

@triton.jit
def ifftshift(input, dim=None):
    """
    Rearranges the elements of the input tensor in FFT order
    such that the zero-frequency component is moved back to the original position.

    Args:
        input: The tensor in FFT order.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2, ...].
        dim: The dimensions to rearrange.
            Only dimensions specified here will be rearranged,
            any other dimensions will be left in their original order.
            Default: All dimensions of input.

    Returns:
        Tensor with elements rearranged.
    """
    # Get the shape of the input tensor
    shape = input.shape
    if dim is None:
        dim = range(len(shape))  # Rearrange all dimensions

    # Create an output tensor
    output = tl.zeros_like(input)

    # Rearranging logic
    for d in dim:
        # Calculate the shift for the current dimension
        shift = shape[d] // 2
        # Rearrange the elements
        output = tl.roll(input, shifts=-shift, axis=d)

    return output
