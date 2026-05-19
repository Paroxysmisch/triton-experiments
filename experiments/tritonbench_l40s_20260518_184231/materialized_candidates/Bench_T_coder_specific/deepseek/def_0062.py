import triton.language as tl

@triton.jit
def scaled_add_dot(y_ptr, x_ptr, alpha_ptr, n, output_ptr):
    # Define the base index for the current thread
    idx = tl.program_id(axis=0)

    # Load the inputs
    y = tl.load(y_ptr + idx)
    x = tl.load(x_ptr + idx)
    alpha = tl.load(alpha_ptr)

    # Perform the scaled addition
    y += alpha * x

    # Compute the dot product of the modified y with itself
    dot_product = tl.dot(y, y)

    # Store the result
    tl.store(output_ptr, dot_product)
