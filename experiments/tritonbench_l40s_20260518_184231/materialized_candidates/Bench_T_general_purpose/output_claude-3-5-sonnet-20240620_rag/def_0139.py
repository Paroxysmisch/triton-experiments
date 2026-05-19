import triton
import triton.language as tl

@triton.jit
def std_kernel(input, dim, correction, keepdim, out, N):
    """
    Kernel to calculate the standard deviation of the input tensor.

    Args:
        input: Input tensor.
        dim: Dimension(s) to reduce.
        correction: Difference between sample size and degrees of freedom.
        keepdim: Whether to retain reduced dimensions.
        out: Output tensor.
        N: Size of the dimension being reduced.
    """
    # Calculate mean
    mean = tl.sum(input, axis=dim) / N
    # Calculate variance
    diff = input - mean
    variance = tl.sum(diff * diff, axis=dim) / (N - correction)
    
    # Calculate standard deviation
    stddev = tl.sqrt(tl.maximum(variance, 0.0))
    
    # Store result in output tensor
    if keepdim:
        out = stddev[:, None]  # Retain dimensions
    else:
        out = stddev  # Remove dimensions

@triton.jit
def std(input, dim=None, *, correction=1, keepdim=False, out=None):
    """
    Wrapper function for standard deviation calculation.

    Args:
        input (Tensor): The input tensor.
        dim (int or tuple of ints): The dimension or dimensions to reduce.
        correction (int): Difference between the sample size and sample degrees of freedom.
        keepdim (bool): Whether the output tensor has dim retained or not.
        out (Tensor, optional): The output tensor.
    
    Returns:
        Tensor: The standard deviation of the input tensor.
    """
    # Determine the dimensions to reduce
    if dim is None:
        dim = tuple(range(input.ndim))  # Reduce over all dimensions
    elif isinstance(dim, int):
        dim = (dim,)  # Convert single dimension to tuple

    # Calculate size of the dimension being reduced
    N = input.shape[dim[0]]  # Assuming dim is a single dimension for simplicity

    # Call the kernel
    std_kernel(input, dim, correction, keepdim, out, N)
