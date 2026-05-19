def min(input, dim, keepdim=False, *, out=None):
    # Define the output types
    output_types = [triton.output.int8(), triton.output.int32()]
    
    # Define the kernel
    kernel = triton.kernel(
        launch_func=lambda input, dim, keepdim, output, output_indices: _min(input, dim, keepdim, output, output_indices),
        inputs=[input, dim, keepdim],
        outputs=[output_types[0], output_types[1]],
        grid=None,
        num_warps=1,
        num_stages=1,
        dynamic_shared_memory=False,
        **triton.profiling.autogen_profiling_info(min)
    )
    
    # Define the output tensors
    output = triton.output.allocate(output_types[0], out=out)
    output_indices = triton.output.allocate(output_types[1], out=out)
    
    # Run the kernel
    kernel(input, dim, keepdim, output, output_indices)
    
    return output, output_indices
