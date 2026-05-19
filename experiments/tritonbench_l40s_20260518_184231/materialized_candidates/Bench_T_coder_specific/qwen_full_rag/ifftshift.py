import triton
import triton.language as tl


@triton.jit
def ifftshift(input, dim=None):
    # Calculate the sizes of each dimension
    sizes = list(input.shape)
    
    # If dim is not specified, shift all dimensions
    if dim is None:
        dim = list(range(len(sizes)))
    
    # Ensure dim is a list for iteration
    if isinstance(dim, int):
        dim = [dim]
    
    # Compute the shift required for each dimension
    shifts = [(size // 2 + size % 2) for size in sizes]
    
    # Create a list of indices for indexing
    indices = list(range(len(sizes)))
    
    # Apply the shift for each dimension specified in dim
    for d in dim:
        indices[d] = (indices[d] + shifts[d]) % sizes[d]
    
    # Return the rearranged tensor according to computed indices
    return tl.take(input, indices)
