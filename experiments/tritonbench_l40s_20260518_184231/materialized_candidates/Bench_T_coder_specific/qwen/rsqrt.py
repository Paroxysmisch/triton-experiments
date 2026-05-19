import triton
import triton.language as tl

# Define the Triton kernel
rsqrt_kernel = triton.compile(rsqrt_kernel)

def rsqrt(input, *, out=None):
    # Check if the input is a valid tensor
    if not isinstance(input, Tensor):
        raise ValueError("Input must be a Tensor")
    
    # Get the shape and dtype of the input tensor
    input_shape = input.shape
    input_dtype = input.dtype
    
    # Create an output tensor if not provided
    if out is None:
        out = Tensor(shape=input_shape, dtype=input_dtype)
    else:
        if out.shape != input_shape or out.dtype != input_dtype:
            raise ValueError("Output tensor shape and dtype must match input tensor")
    
    # Get the total number of elements in the tensor
    N = np.prod(input_shape)
    
    # Launch the Triton kernel
    rsqrt_kernel[(N + BLOCK_SIZE - 1) // BLOCK_SIZE](input.data_ptr(), out.data_ptr(), N, BLOCK_SIZE=BLOCK_SIZE)
    
    return out
