{{ code }}
import triton
import triton.language as tl

@triton.jit
def ifftshift_kernel(input_ptr, output_ptr, shape, dim, n_elements):
    # Calculate the index for the input tensor
    idx = tl.arange(0, n_elements)
    
    # Determine the number of dimensions
    num_dims = len(shape)
    
    # Calculate the rearrangement based on the specified dimensions
    for d in range(num_dims):
        if dim is None or d in dim:
            # Calculate the shift for the current dimension
            shift = shape[d] // 2
            # Rearrange the indices
            idx = (idx + shift) % shape[d]
    
    # Load input tensor and store in output tensor
    output_ptr[idx] = input_ptr[idx]

def ifftshift(input, dim=None):
    # Get the shape of the input tensor
    shape = input.shape
    n_elements = input.numel()
    
    # Allocate output tensor
    output = torch.empty_like(input)
    
    # Launch the Triton kernel
    ifftshift_kernel[(n_elements,)](input, output, shape, dim, n_elements)
    
    return output
{{ code }}
