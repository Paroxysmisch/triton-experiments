import triton
import triton.language as tl

@triton.jit
def grid_sample_kernel(
    input_ptr,  # Pointer to the input tensor
    grid_ptr,   # Pointer to the grid tensor
    output_ptr, # Pointer to the output tensor
    input_shape,  # Shape of the input tensor (N, C, H_in, W_in) or (N, C, D_in, H_in, W_in)
    grid_shape,   # Shape of the grid tensor (N, H_out, W_out, 2) or (N, D_out, H_out, W_out, 3)
    output_shape, # Shape of the output tensor (N, C, H_out, W_out) or (N, C, D_out, H_out, W_out)
    mode,         # Interpolation mode: 'bilinear' or 'nearest'
    padding_mode, # Padding mode: 'zeros', 'border', 'reflection'
    align_corners,  # Whether to align corners
    BLOCK_SIZE: tl.constexpr
):
    # Get the current block index
    pid = tl.program_id(axis=0)
    # Compute the number of elements to process
    n_elements = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]
    if len(output_shape) == 5:
        n_elements *= output_shape[4]
    # Compute the starting and ending indices for this block
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, n_elements)
    
    # Iterate over the elements in this block
    for i in range(block_start, block_end):
        # Compute the indices for the output tensor
        n = i // (output_shape[1] * output_shape[2] * output_shape[3])
        c = (i % (output_shape[1] * output_shape[2] * output_shape[3])) // (output_shape[2] * output_shape[3])
        h_out = (i % (output_shape[2] * output_shape[3])) // output_shape[3]
        w_out = i % output_shape[3]
        d_out = 0
        if len(output_shape) == 5:
            d_out = h_out
            h_out = (i % (output_shape[3] * output_shape[4])) // output_shape[4]
            w_out = i % output_shape[4]
        
        # Compute the grid indices
        grid_idx = n * grid_shape[1] * grid_shape[2] * grid_shape[3] + h_out * grid_shape[2] * grid_shape[3] + w_out * grid_shape[3] + d_out * grid_shape[3] + c
        grid_x = tl.load(grid_ptr + grid_idx * 2 + 0)
        grid_y = tl.load(grid_ptr + grid_idx * 2 + 1)
        grid_z = 0.0
        if len(grid_shape) == 5:
            grid_z = tl.load(grid_ptr + grid_idx * 3 + 2)
        
        # Handle NaN values in the grid
        grid_x = tl.where(tl.isnan(grid_x), -1.0, grid_x)
        grid_y = tl.where(tl.isnan(grid_y), -1.0, grid_y)
        grid_z = tl.where(tl.isnan(grid_z), -1.0, grid_z)
        
        # Normalize grid values to the input tensor dimensions
        if align_corners:
            grid_x = (grid_x + 1) * (input_shape[3] - 1) / 2
            grid_y = (grid_y + 1) * (input_shape[2] - 1) / 2
            grid_z = (grid_z + 1) * (input_shape[4] - 1) / 2
        else:
            grid_x = (grid_x + 1) * (input_shape[3] - 1) / 2
            grid_y = (grid_y + 1) * (input_shape[2] - 1) / 2
            grid_z = (grid_z + 1) * (input_shape[4] - 1) / 2
        
        # Compute the output value using the specified interpolation mode
        if mode == 'nearest':
            x = tl.max(tl.min(tl.round(grid_x).to(tl.int32), input_shape[3] - 1), 0)
            y = tl.max(tl.min(tl.round(grid_y).to(tl.int32), input_shape[2] - 1), 0)
            z = 0
            if len(input_shape) == 5:
                z = tl.max(tl.min(tl.round(grid_z).to(tl.int32), input_shape[4] - 1), 0)
            input_idx = n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + y * input_shape[3] + x + z * input_shape[2] * input_shape[3]
            output_val = tl.load(input_ptr + input_idx)
        elif mode == 'bilinear':
            x0 = tl.max(tl.min(tl.floor(grid_x).to(tl.int32), input_shape[3] - 2), 0)
            x1 = x0 + 1
            y0 = tl.max(tl.min(tl.floor(grid_y).to(tl.int32), input_shape[2] - 2), 0)
            y1 = y0 + 1
            z0 = 0
            z1 = 0
            if len(input_shape) == 5:
                z0 = tl.max(tl.min(tl.floor(grid_z).to(tl.int32), input_shape[4] - 2), 0)
                z1 = z0 + 1
            w00 = (x1 - grid_x) * (y1 - grid_y)
            w01 = (x1 - grid_x) * (grid_y - y0)
            w10 = (grid_x - x0) * (y1 - grid_y)
            w11 = (grid_x - x0) * (grid_y - y0)
            w000 = w00
            w001 = w00
            w010 = w01
            w011 = w01
            w100 = w10
            w101 = w10
            w110 = w11
            w111 = w11
            if len(input_shape) == 5:
                w000 = w00 * (z1 - grid_z)
                w001 = w00 * (grid_z - z0)
                w010 = w01 * (z1 - grid_z)
                w011 = w01 * (grid_z - z0)
                w100 = w10 * (z1 - grid_z)
                w101 = w10 * (grid_z - z0)
                w110 = w11 * (z1 - grid_z)
                w111 = w11 * (grid_z - z0)
            input_idx000 = n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + y0 * input_shape[3] + x0 + z0 * input_shape[2] * input_shape[3]
            input_idx001 = n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + y0 * input_shape[3] + x0 + z1 * input_shape[2] * input_shape[3]
            input_idx010 = n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + y0 * input_shape[3] + x1 + z0 * input_shape[2] * input_shape[3]
            input_idx011 = n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + y0 * input_shape[3] + x1 + z1 * input_shape[2] * input_shape[3]
            input_idx100 = n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + y1 * input_shape[3] + x0 + z0 * input_shape[2] * input_shape[3]
            input_idx101 = n * input_shape[1] * input_shape[2] * input_shape[3] + c * input_shape[2] * input_shape[3] + y1 * input_shape[3] + x0 + z1 * input_shape[2] * input_shape[3]
            input_idx110 = n * input_shape[1] * input_shape[2] * input
