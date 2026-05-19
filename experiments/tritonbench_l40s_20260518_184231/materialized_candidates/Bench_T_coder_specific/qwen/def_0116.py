import triton
import triton.language as tl

@triton.jit
def sum_kernel(
    x_ptr,           # *ptr to the input tensor
    y_ptr,           # *ptr to the output tensor
    n_elements,      # number of elements to reduce
    n_reduced_dims,  # number of reduced dimensions
    stride_per_dim,  # stride per dimension
    block_size,      # block size along the reduced dimension
):
    # Determine the global index within the entire tensor
    pid = tl.program_id(axis=0)
    
    # Compute the linear offset for the current element
    x_offset = pid * n_elements
    
    # Initialize the reduction result for the current thread
    reduction_result = tl.zeros([], dtype=tl.float32)
    
    # Iterate over the elements in the block
    for i in range(block_size):
        # Load the current element from the input tensor
        x_val = tl.load(x_ptr + x_offset + i * stride_per_dim[pid])
        
        # Perform the reduction operation
        reduction_result += x_val
    
    # Store the reduction result in the output tensor
    tl.store(y_ptr + pid, reduction_result)

# Function to launch the Triton kernel
def launch_sum_kernel(x, y, dim, keepdim):
    # Get the shape of the input tensor
    x_shape = x.shape
    
    # Calculate the strides per dimension
    stride_per_dim = [x_shape[i] for i in range(len(x_shape))]
    
    # Calculate the number of reduced dimensions
    n_reduced_dims = len(dim)
    
    # Calculate the total number of elements to reduce
    n_elements = 1
    for d in dim:
        n_elements *= x_shape[d]
    
    # Calculate the block size along the reduced dimension
    block_size = 1024 // n_reduced_dims
    
    # Launch the Triton kernel
    grid_size = (n_elements + block_size - 1) // block_size
    triton.launch_kernel(sum_kernel, (grid_size,), {
        "x_ptr": x.data_ptr(),
        "y_ptr": y.data_ptr(),
        "n_elements": n_elements,
        "n_reduced_dims": n_reduced_dims,
        "stride_per_dim": stride_per_dim,
        "block_size": block_size,
    })
