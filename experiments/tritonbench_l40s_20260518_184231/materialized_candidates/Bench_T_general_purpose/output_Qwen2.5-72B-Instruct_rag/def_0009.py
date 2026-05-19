import torch
import triton
import triton.language as tl

# Triton kernel for grid sampling
@triton.jit
def grid_sample_kernel(
    input_ptr,  # Pointer to the input tensor in GPU memory
    grid_ptr,  # Pointer to the grid tensor in GPU memory
    output_ptr,  # Pointer to the output tensor in GPU memory
    input_shape,  # Shape of the input tensor (N, C, H, W) or (N, C, D, H, W)
    grid_shape,  # Shape of the grid tensor (N, H_out, W_out, 2) or (N, D_out, H_out, W_out, 3)
    output_shape,  # Shape of the output tensor (N, C, H_out, W_out) or (N, C, D_out, H_out, W_out)
    mode,  # Interpolation mode: 'bilinear', 'nearest', 'bicubic'
    padding_mode,  # Padding mode: 'zeros', 'border', 'reflection'
    align_corners,  # Whether to align corners
    BLOCK_SIZE: tl.constexpr  # Size of each block of threads
):
    pid = tl.program_id(axis=0)  # Get the block index
    batch, channels, *input_spatial = input_shape
    batch, *grid_spatial, _ = grid_shape
    batch, channels, *output_spatial = output_shape
    n_elements = batch * channels * volume(output_spatial)
    block_start = pid * BLOCK_SIZE  # Calculate the start index for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Calculate offsets for each thread
    mask = offsets < n_elements  # Mask to ensure we don't write out of bounds

    # Flatten the output tensor indices
    flat_indices = tl.arange(0, BLOCK_SIZE) + block_start
    flat_indices = flat_indices[mask]

    # Unflatten the output tensor indices
    batch_indices = flat_indices // (channels * volume(output_spatial))
    channel_indices = (flat_indices % (channels * volume(output_spatial))) // volume(output_spatial)
    spatial_indices = flat_indices % volume(output_spatial)

    # Compute the grid indices
    grid_indices = tl.zeros((BLOCK_SIZE, len(grid_spatial)), dtype=tl.int32)
    for i in range(len(grid_spatial)):
        grid_indices[:, i] = spatial_indices // volume(output_spatial[i+1:])
        spatial_indices = spatial_indices % volume(output_spatial[i+1:])

    # Load the grid values
    grid_values = tl.load(grid_ptr + batch_indices * volume(grid_spatial) + tl.arange(0, len(grid_spatial)) * volume(grid_spatial[1:]), mask=mask)

    # Handle NaN values in the grid
    grid_values = tl.where(tl.isnan(grid_values), -1.0, grid_values)

    # Normalize grid values to [-1, 1] and convert to input tensor indices
    input_indices = tl.zeros((BLOCK_SIZE, len(grid_spatial)), dtype=tl.float32)
    for i in range(len(grid_spatial)):
        if align_corners:
            input_indices[:, i] = ((grid_values[:, i] + 1) / 2) * (input_spatial[i] - 1)
        else:
            input_indices[:, i] = ((grid_values[:, i] + 1) / 2) * input_spatial[i] - 0.5

    # Perform interpolation
    if mode == 'nearest':
        input_indices = tl.round(input_indices).to(tl.int32)
        input_indices = tl.where(input_indices < 0, 0, input_indices)
        input_indices = tl.where(input_indices >= input_spatial, input_spatial - 1, input_indices)
        input_indices = input_indices + batch_indices * volume(input_spatial) + channel_indices * volume(input_spatial)
        output_values = tl.load(input_ptr + input_indices, mask=mask)
    elif mode == 'bilinear':
        # Bilinear interpolation
        x0 = tl.floor(input_indices[:, 0]).to(tl.int32)
        x1 = x0 + 1
        y0 = tl.floor(input_indices[:, 1]).to(tl.int32)
        y1 = y0 + 1

        x0 = tl.where(x0 < 0, 0, x0)
        x1 = tl.where(x1 >= input_spatial[0], input_spatial[0] - 1, x1)
        y0 = tl.where(y0 < 0, 0, y0)
        y1 = tl.where(y1 >= input_spatial[1], input_spatial[1] - 1, y1)

        q11 = tl.load(input_ptr + batch_indices * volume(input_spatial) + channel_indices * volume(input_spatial) + x0 * input_spatial[1] + y0, mask=mask)
        q12 = tl.load(input_ptr + batch_indices * volume(input_spatial) + channel_indices * volume(input_spatial) + x0 * input_spatial[1] + y1, mask=mask)
        q21 = tl.load(input_ptr + batch_indices * volume(input_spatial) + channel_indices * volume(input_spatial) + x1 * input_spatial[1] + y0, mask=mask)
        q22 = tl.load(input_ptr + batch_indices * volume(input_spatial) + channel_indices * volume(input_spatial) + x1 * input_spatial[1] + y1, mask=mask)

        wa = (x1 - input_indices[:, 0]) * (y1 - input_indices[:, 1])
        wb = (x1 - input_indices[:, 0]) * (input_indices[:, 1] - y0)
        wc = (input_indices[:, 0] - x0) * (y1 - input_indices[:, 1])
        wd = (input_indices[:, 0] - x0) * (input_indices[:, 1] - y0)

        output_values = wa * q11 + wb * q12 + wc * q21 + wd * q22
    elif mode == 'bicubic':
        # Bicubic interpolation (simplified for brevity)
        raise NotImplementedError("Bicubic interpolation is not implemented in this kernel.")

    # Apply padding mode
    if padding_mode == 'zeros':
        output_values = tl.where((input_indices < 0) | (input_indices >= input_spatial), 0.0, output_values)
    elif padding_mode == 'border':
        input_indices = tl.where(input_indices < 0, 0, input_indices)
        input_indices = tl.where(input_indices >= input_spatial, input_spatial - 1, input_indices)
    elif padding_mode == 'reflection':
        input_indices = tl.where(input_indices < 0, -2 * input_indices, input_indices)
        input_indices = tl.where(input_indices >= input_spatial, 2 * input_spatial - 2 - input_indices, input_indices)

    # Store the output values
    tl.store(output_ptr + flat_indices, output_values, mask=mask)

# Function to perform grid sampling using the Triton kernel
def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False) -> torch.Tensor:
    if input.dim() not in [4, 5]:
        raise ValueError("Input tensor must be 4D (N, C, H, W) or 5D (N, C, D, H, W).")
    if grid.dim() not in [4, 5]:
        raise ValueError("Grid tensor must be 4D (N, H_out, W_out, 2) or 5D (N, D_out, H_out, W_out, 3).")

    input_shape = input.shape
    grid_shape = grid.shape
    output_shape = input_shape[:2] + grid_shape[1:-1]

    output = torch.empty(output_shape, device=input.device, dtype=input.dtype)

    n_elements = volume(output_shape)
    grid_fn = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch.cuda.device(input.device):
        grid_sample_kernel[grid_fn](
            input.data_ptr(),
            grid.data_ptr(),
            output.data_ptr(),
            input_shape,
            grid_shape,
            output_shape,
            mode,
            padding_mode,
            align_corners,
            BLOCK_SIZE=1024
        )

    return output

# Helper function to compute the volume of a tensor shape
def volume(shape):
    return torch.prod(torch.tensor(shape)).item()
