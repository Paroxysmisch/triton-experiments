import triton
import triton.language as tl

@triton.jit
def gelu_kernel(input_ptr, output_ptr, n_elements, approximate, BLOCK_SIZE: tl.constexpr):
    # Create a block of indices
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load input data
    x = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)

    # Compute GELU
    if approximate == 0:  # exact
        sqrt_2_over_pi = 0.7978845608  # sqrt(2/pi)
        cdf = 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        gelu = x * cdf
    else:  # approximate using tanh
        sqrt_2_over_pi = 0.7978845608  # sqrt(2/pi)
        x_cubed = x * x * x
        tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        gelu = 0.5 * x * (1.0 + tl.tanh(tanh_arg))

    # Store the result
    tl.store(output_ptr + offsets, gelu, mask=offsets < n_elements)

def gelu(input, approximate='none'):
    # Determine the size of the input tensor
    n_elements = input.numel()
    
    # Allocate output tensor
    output = triton.empty_like(input)
    
    # Determine the approximation mode
    approximate_mode = 0 if approximate == 'none' else 1
    
    # Define block size for Triton kernel
    BLOCK_SIZE = 1024
    
    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](input, output, n_elements, approximate_mode, BLOCK_SIZE=BLOCK_SIZE)
    
    return output
