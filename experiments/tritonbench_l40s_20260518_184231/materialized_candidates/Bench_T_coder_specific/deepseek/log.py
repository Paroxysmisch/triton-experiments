import triton.language as tl

@triton.jit
def log_kernel(input_ptr, output_ptr, n):
    # Define the loop index
    i = tl.program_id(0)

    # Load the input value
    x = tl.load(input_ptr + i)

    # Compute the natural logarithm
    y = tl.log(x)

    # Store the result
    tl.store(output_ptr + i, y)

def log(input, out=None):
    # Get the number of elements
    n = input.numel()

    # Get the input and output pointers
    input_ptr = triton.pointers.allocate_input(input, n)
    output_ptr = triton.pointers.allocate_output(out, n)

    # Run the kernel
    log_kernel[n](input_ptr, output_ptr, n)

    # Return the output
    return output_ptr
