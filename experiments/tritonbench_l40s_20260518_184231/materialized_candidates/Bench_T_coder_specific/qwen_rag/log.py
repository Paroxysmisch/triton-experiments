import triton
import triton.language as tl

@triton.jit
def log_kernel(output_ptr, input_ptr, n_elements: tl.constexpr):
    # Each thread processes one element
    idx = tl.program_id(0)
    
    # Load the input element
    input_val = tl.load(input_ptr + idx)
    
    # Compute the natural logarithm
    output_val = tl.math.log(input_val)
    
    # Store the result
    tl.store(output_ptr + idx, output_val)
