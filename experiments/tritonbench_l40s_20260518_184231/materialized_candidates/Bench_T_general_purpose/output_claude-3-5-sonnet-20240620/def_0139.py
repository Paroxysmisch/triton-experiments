import triton
import triton.language as tl

@triton.jit
def std_kernel(input_ptr, output_ptr, N, dim, correction, keepdim):
    # Calculate the mean
    mean = tl.zeros((1,), dtype=tl.float32)
    for i in range(N):
        mean += input_ptr[i]
    mean /= N

    # Calculate the variance
    variance = tl.zeros((1,), dtype=tl.float32)
    for i in range(N):
        variance += (input_ptr[i] - mean) ** 2
    variance /= max(1, N - correction)

    # Calculate the standard deviation
    stddev = tl.sqrt(variance)

    # Store the result
    if keepdim:
        output_ptr[0] = stddev
    else:
        output_ptr[0] = stddev[0]  # Remove dimensions if not keeping them

def std(input, dim=None, *, correction=1, keepdim=False, out=None) -> Tensor:
    # Validate input tensor
    if not isinstance(input, Tensor):
        raise TypeError("Input must be a Tensor.")
    
    # Determine the number of elements in the input tensor
    N = input.numel()
    
    # Prepare output tensor
    if out is None:
        out = input.new_zeros((1,), dtype=input.dtype)
    
    # Call the Triton kernel
    std_kernel[(1,)](input.data_ptr(), out.data_ptr(), N, dim, correction, keepdim)
    
    return out
