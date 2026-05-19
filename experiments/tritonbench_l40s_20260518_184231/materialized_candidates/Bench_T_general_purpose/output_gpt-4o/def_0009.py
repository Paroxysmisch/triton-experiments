import triton
import triton.language as tl

@triton.jit
def grid_sample_kernel(input_ptr, grid_ptr, output_ptr, B, C, H, W, mode, padding_mode, align_corners, stride_in, stride_out, stride_grid):
    pid = tl.program_id(0)
    n, c, h, w = tl.program_id(0), tl.program_id(1), tl.program_id(2), tl.program_id(3)

    # Calculate grid positions
    grid_x = tl.load(grid_ptr + n * stride_grid[0] + h * stride_grid[1] + w * stride_grid[2] + 0)
    grid_y = tl.load(grid_ptr + n * stride_grid[0] + h * stride_grid[1] + w * stride_grid[2] + 1)

    # Handle NaN in grid
    grid_x = tl.where(tl.isnan(grid_x), -1.0, grid_x)
    grid_y = tl.where(tl.isnan(grid_y), -1.0, grid_y)

    # Normalize grid to input dimensions
    if align_corners:
        grid_x = (grid_x + 1) * 0.5 * (W - 1)
        grid_y = (grid_y + 1) * 0.5 * (H - 1)
    else:
        grid_x = (grid_x + 1) * W * 0.5
        grid_y = (grid_y + 1) * H * 0.5

    # Calculate pixel locations
    x0 = tl.floor(grid_x).to(tl.int32)
    x1 = x0 + 1
    y0 = tl.floor(grid_y).to(tl.int32)
    y1 = y0 + 1

    # Interpolation weights
    if mode == 'bilinear':
        wx = grid_x - x0
        wy = grid_y - y0
    else:  # nearest
        wx = wy = 0.0

    # Fetch values and apply padding mode
    def fetch_value(x, y):
        if padding_mode == 'zeros':
            cond_x = (0 <= x) & (x < W)
            cond_y = (0 <= y) & (y < H)
            return tl.where(cond_x & cond_y, tl.load(input_ptr + n * stride_in[0] + c * stride_in[1] + y * stride_in[2] + x * stride_in[3]), 0.0)
        else:
            # Handle other padding modes if needed
            return 0.0

    # Perform interpolation
    v00 = fetch_value(x0, y0)
    v01 = fetch_value(x0, y1)
    v10 = fetch_value(x1, y0)
    v11 = fetch_value(x1, y1)

    if mode == 'bilinear':
        output_value = (1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v10 + (1 - wx) * wy * v01 + wx * wy * v11
    else:  # nearest
        output_value = v00

    # Store the result
    tl.store(output_ptr + n * stride_out[0] + c * stride_out[1] + h * stride_out[2] + w * stride_out[3], output_value)

import torch

def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    # Validate inputs
    assert mode in ['bilinear', 'nearest'], "Unsupported mode"
    assert padding_mode in ['zeros'], "Unsupported padding_mode"  # Extend if more modes are implemented

    B, C, H, W = input.shape
    output = torch.empty_like(input)

    # Strides for memory access
    stride_in = input.stride()
    stride_out = output.stride()
    stride_grid = grid.stride()

    # Launch the Triton kernel
    grid_sample_kernel[(B, C, H, W)](
        input_ptr=input,
        grid_ptr=grid,
        output_ptr=output,
        B=B, C=C, H=H, W=W,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
        stride_in=stride_in,
        stride_out=stride_out,
        stride_grid=stride_grid
    )

    return output
