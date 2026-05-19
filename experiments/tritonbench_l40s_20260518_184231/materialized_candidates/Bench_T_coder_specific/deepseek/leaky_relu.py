def leaky_relu(input, negative_slope=0.01, inplace=False):
    import triton.language as tl

    @triton.jit
    def leaky_relu_kernel(input_ptr, output_ptr, num_elements, negative_slope):
        # Define the loop index
        pid = tl.program_id(axis=0)
        # Compute the number of threads in the grid
        num_warps = tl.cdiv(num_elements, tl.warps_per_cta(axis=0))
        # Define the loop index
        warp_id = pid // tl.warp_size
        # Define the element index
        element_id = warp_id * tl.warp_size + tl.local_id(axis=0)
        # Load the input element
        element = tl.load(input_ptr + element_id)
        # Apply the LeakyReLU activation function
        output = tl.max(element, 0) + negative_slope * tl.min(element, 0)
        # Store the output element
        tl.store(output_ptr + element_id, output)

    # Allocate memory for the output tensor
    output = tl.empty_like(input)
    # Compute the number of elements
    num_elements = output.num_elements
    # Compute the number of threads in the grid
    num_warps = tl.cdiv(num_elements, tl.warps_per_cta(axis=0))
    # Launch the kernel
    leaky_relu_kernel[num_warps, tl.threads_per_warp](input, output, num_elements, negative_slope)
    # Return the output tensor
    return output
