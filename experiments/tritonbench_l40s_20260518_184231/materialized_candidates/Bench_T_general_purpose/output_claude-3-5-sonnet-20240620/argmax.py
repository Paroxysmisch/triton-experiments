import triton
import triton.language as tl

@triton.jit
def argmax_kernel(input_ptr, output_ptr, dim, keepdim, n_elements):
    # Get the thread's unique index
    idx = tl.program_id(0)
    
    # Ensure we are within bounds
    if idx >= n_elements:
        return

    # Load the input tensor
    input_tensor = tl.load(input_ptr + idx)

    # Initialize max value and index
    max_value = -tl.inf
    max_index = -1

    # Iterate over the specified dimension
    for i in range(dim):
        value = tl.load(input_ptr + idx + i)
        if value > max_value:
            max_value = value
            max_index = i

    # Store the result
    tl.store(output_ptr + idx, max_index)

def argmax(input, dim, keepdim=False):
    # Flatten the input tensor if dim is None
    if dim is None:
        dim = 0  # Set to 0 for flattening

    # Get the number of elements
    n_elements = input.numel()

    # Allocate output tensor
    output = torch.empty(n_elements, dtype=torch.long)

    # Launch the kernel
    argmax_kernel[(n_elements,)](input.data_ptr(), output.data_ptr(), dim, keepdim, n_elements)

    # If keepdim is True, reshape the output tensor
    if keepdim:
        output = output.view(-1, 1)

    return output
