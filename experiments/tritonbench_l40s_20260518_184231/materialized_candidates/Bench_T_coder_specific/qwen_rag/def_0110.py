import triton
import triton.language as tl

@triton.jit
def exp_mean_kernel(input_ptr, output_ptr, input_shape, output_shape, stride, dim, block_size: tl.constexpr):
    # Determine the grid and block indices
    x = tl.program_id(axis=0)
    
    # Calculate the base index for the current thread
    base_idx = x * block_size
    
    # Initialize the sum and count
    exp_sum = tl.zeros([], dtype=input_ptr.dtype)
    count = tl.zeros([], dtype=input_ptr.dtype)
    
    # Iterate over the elements in the specified dimension
    for i in range(block_size):
        idx = base_idx + i
        
        # Check if the index is within bounds
        if idx < input_shape[dim]:
            # Apply the exponential function
            exp_value = tl.exp(input_ptr[idx])
            
            # Add the exponential value to the sum
            exp_sum += exp_value
            
            # Increment the count
            count += 1
    
    # Reduce the sum and count across threads in the block
    exp_sum = tl.reduce(exp_sum, axis=0, op=tl.add)
    count = tl.reduce(count, axis=0, op=tl.add)
    
    # Compute the mean
    mean = exp_sum / count
    
    # Store the result in the output tensor
    output_ptr[x] = mean

# Wrapper Function
@triton.jit
def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None) -> tl.Tensor:
    # Get the shape of the input tensor
    input_shape = input.shape
    
    # Determine the output shape
    if dim is None:
        output_shape = ()
    else:
        output_shape = tuple([size for i, size in enumerate(input_shape) if i != dim])
        if keepdim:
            output_shape = output_shape + (1,)
    
    # Allocate memory for the output tensor
    if out is None:
        out = tl.zeros(output_shape, dtype=dtype if dtype is not None else input.dtype)
    else:
        assert out.shape == output_shape, "Output shape mismatch"
        assert out.dtype == dtype if dtype is not None else input.dtype, "Output dtype mismatch"
    
    # Set the block size
    block_size = 1024
    
    # Launch the kernel
    grid_size = (len(out),)
    exp_mean_kernel[grid_size, block_size](input.data, out.data, input_shape, output_shape, input.stride, dim, block_size)
    
    return out
