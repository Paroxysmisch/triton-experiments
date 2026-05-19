import triton
import triton.language as tl

@triton.jit
def fused_gather_masked_fill_kernel(
    X_ptr, dim, I_ptr, M_ptr, value, Y_ptr,
    X_shape, I_shape, M_shape, Y_shape,
    block_size=1024):
    
    # Get the global thread ID within the block
    pid = tl.program_id(axis=0)
    coords = tl.program_id(axis=1)
    
    # Compute the offset for the current thread
    x_offset = pid * block_size + coords
    
    # Iterate over the batch dimension
    for i in range(Y_shape[0]):
        # Compute the base coordinates
        base_coords = [i]
        
        # Iterate over all dimensions
        for d in range(len(X_shape)):
            if d == dim:
                continue
            # Compute the coordinate for this dimension
            coord = x_offset % X_shape[d]
            base_coords.append(coord)
            x_offset //= X_shape[d]
        
        # Compute the index for gathering
        index_coord = base_coords[:]
        index_coord[dim] = I_ptr[i, tuple(base_coords)]
        
        # Gather the value
        y_value = X_ptr[i, tuple(index_coord)]
        
        # Apply the mask
        if M_ptr[i, tuple(base_coords)]:
            y_value = value
        
        # Store the result
        Y_ptr[i, tuple(base_coords)] = y_value

# Define the grid and block sizes
block_size = 1024
grid_size = lambda x, y: ((x + block_size - 1) // block_size, (y + block_size - 1) // block_size)

# Launch the kernel
fused_gather_masked_fill_kernel[X_shape[0], block_size](X_ptr, dim, I_ptr, M_ptr, value, Y_ptr, X_shape, I_shape, M_shape, Y_shape)
