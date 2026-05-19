triton
import triton
import triton.language as tl

@triton.jit
def sub_gelu_kernel(
    input_ptr, other_ptr, alpha_ptr, output_ptr,
    N, BLOCK_SIZE: tl.constexpr):
    
    # Get the index of the current thread
    pid = tl.program_id(axis=0)
    num_blocks = tl.cdiv(N, BLOCK_SIZE)
    grid_size = num_blocks
    
    # Ensure we don't go out of bounds
    if pid >= grid_size:
        return
    
    # Load data into shared memory
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(input_ptr + i)
    y = tl.load(other_ptr + i)
    a = tl.load(alpha_ptr + i)
    
    # Perform the subtraction and scaling
    z = x - a * y
    
    # Apply GELU
    if tl.constexpr(approximate == 'none'):
        gelu_result = z * tl.nn.gelu(z)
    elif tl.constexpr(approximate == 'tanh'):
        sqrt_two_over_pi = 0.7978845608
        c = 0.044715
        tanh_approx = 0.5 * (1 + tl.tanh(sqrt_two_over_pi * (z + c * z ** 3)))
        gelu_result = 0.5 * z * tanh_approx
    else:
        raise ValueError("Invalid approximate value")
    
    # Store the result back to global memory
    tl.store(output_ptr + i, gelu_result)

# Wrapper Function
def sub_gelu(input, other, alpha=1, approximate='none', out=None):
    # Check input shapes
    assert input.shape == other.shape, "Input and other must have the same shape"
    
    # Determine the size of the input
    N = input.size
    
    # Create a default output tensor if none is provided
    if out is None:
        out = input.new_empty(input.shape)
    
    # Set up Triton grid and block sizes
    block_size = 256
    grid_size = (N + block_size - 1) // block_size
    
    # Launch the Triton kernel
    sub_gelu_kernel[grid_size, block_size](input.data_ptr(), other.data_ptr(), 
                                          tl.zeros_like(input).data_ptr() if alpha == 1 else alpha.data_ptr(),
                                          out.data_ptr(), N, block_size)
    
    return out
