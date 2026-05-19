import torch
import triton
import triton.language as tl

@triton.jit
def grid_sample_kernel(
    input_ptr, grid_ptr, output_ptr,
    input_N, input_C, input_D, input_H, input_W,
    grid_N, grid_D, grid_H, grid_W, grid_C,
    output_N, output_C, output_D, output_H, output_W,
    mode, padding_mode, align_corners,
    stride_input_n, stride_input_c, stride_input_d, stride_input_h, stride_input_w,
    stride_grid_n, stride_grid_d, stride_grid_h, stride_grid_w, stride_grid_c,
    stride_output_n, stride_output_c, stride_output_d, stride_output_h, stride_output_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Determine input dimensionality (4D or 5D)
    is_volumetric = input_D > 0  # 5D if input_D > 0, else 4D

    # 3D launch grid for 5D (N, D, H, W) or 2D for 4D (N, H, W)
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1) if is_volumetric else 0
    pid_h = tl.program_id(2) if is_volumetric else tl.program_id(1)
    pid_w = tl.program_id(3) if is_volumetric else tl.program_id(2)

    # Check bounds
    if pid_n >= input_N:
        return
    if is_volumetric:
        if pid_d >= output_D or pid_h >= output_H or pid_w >= output_W:
            return
    else:
        if pid_h >= output_H or pid_w >= output_W:
            return

    # Get grid values (x, y, [z])
    if is_volumetric:
        grid_idx = (pid_n * stride_grid_n + 
                    pid_d * stride_grid_d + 
                    pid_h * stride_grid_h + 
                    pid_w * stride_grid_w)
        x = tl.load(grid_ptr + grid_idx + 0 * stride_grid_c)
        y = tl.load(grid_ptr + grid_idx + 1 * stride_grid_c)
        z = tl.load(grid_ptr + grid_idx + 2 * stride_grid_c)
    else:
        grid_idx = (pid_n * stride_grid_n + 
                    pid_h * stride_grid_h + 
                    pid_w * stride_grid_w)
        x = tl.load(grid_ptr + grid_idx + 0 * stride_grid_c)
        y = tl.load(grid_ptr + grid_idx + 1 * stride_grid_c)
        z = 0.0  # dummy for 4D

    # Replace NaN with -1
    x = tl.where(tl.math.isnan(x), -1.0, x)
    y = tl.where(tl.math.isnan(y), -1.0, y)
    if is_volumetric:
        z = tl.where(tl.math.isnan(z), -1.0, z)

    # Unnormalize coordinates
    if align_corners:
        ix = (x + 1.0) * (input_W - 1.0) / 2.0
        iy = (y + 1.0) * (input_H - 1.0) / 2.0
        if is_volumetric:
            iz = (z + 1.0) * (input_D - 1.0) / 2.0
    else:
        ix = ((x + 1.0) * input_W - 1.0) / 2.0
        iy = ((y + 1.0) * input_H - 1.0) / 2.0
        if is_volumetric:
            iz = ((z + 1.0) * input_D - 1.0) / 2.0

    # Handle padding mode
    def _coordinate_check(coord, size, padding_mode):
        if padding_mode == 'zeros':
            return coord
        elif padding_mode == 'border':
            return tl.minimum(tl.maximum(coord, 0.0), size - 1.0)
        elif padding_mode == 'reflection':
            coord = tl.abs(coord) % (2.0 * (size - 1.0))
            coord = tl.minimum(coord, 2.0 * (size - 1.0) - coord)
            return coord
        else:
            return coord  # Default to zeros if unknown

    ix = _coordinate_check(ix, input_W, padding_mode)
    iy = _coordinate_check(iy, input_H, padding_mode)
    if is_volumetric:
        iz = _coordinate_check(iz, input_D, padding_mode)

    # Interpolation
    if mode == 'nearest':
        ix_nearest = tl.math.round(ix)
        iy_nearest = tl.math.round(iy)
        if is_volumetric:
            iz_nearest = tl.math.round(iz)
            # Load input value
            val = 0.0
            if (0 <= ix_nearest < input_W and 
                0 <= iy_nearest < input_H and 
                0 <= iz_nearest < input_D):
                input_idx = (pid_n * stride_input_n +
                             tl.math.floor(iz_nearest) * stride_input_d +
                             tl.math.floor(iy_nearest) * stride_input_h +
                             tl.math.floor(ix_nearest) * stride_input_w)
                val = tl.load(input_ptr + input_idx)
        else:
            # 4D case
            val = 0.0
            if (0 <= ix_nearest < input_W and 
                0 <= iy_nearest < input_H):
                input_idx = (pid_n * stride_input_n +
                             tl.math.floor(iy_nearest) * stride_input_h +
                             tl.math.floor(ix_nearest) * stride_input_w)
                val = tl.load(input_ptr + input_idx)
    else:
        # Simplified bilinear interpolation for demonstration
        ix_floor = tl.math.floor(ix)
        iy_floor = tl.math.floor(iy)
        dx = ix - ix_floor
        dy = iy - iy_floor

        # Load input values
        val = 0.0
        for i in range(2):
            for j in range(2):
                x_idx = ix_floor + i
                y_idx = iy_floor + j
                if 0 <= x_idx < input_W and 0 <= y_idx < input_H:
                    input_idx = (pid_n * stride_input_n +
                                 tl.math.floor(y_idx) * stride_input_h +
                                 tl.math.floor(x_idx) * stride_input_w)
                    weight = (1 - dx + i * (2*dx -1)) * (1 - dy + j * (2*dy -1))
                    val += tl.load(input_ptr + input_idx) * weight

    # Write output
    if is_volumetric:
        output_idx = (pid_n * stride_output_n +
                      pid_d * stride_output_d +
                      pid_h * stride_output_h +
                      pid_w * stride_output_w)
    else:
        output_idx = (pid_n * stride_output_n +
                      pid_h * stride_output_h +
                      pid_w * stride_output_w)
    tl.store(output_ptr + output_idx, val)

def grid_sample(input: torch.Tensor, grid: torch.Tensor, mode='bilinear', padding_mode='zeros', align_corners=False) -> torch.Tensor:
    assert input.dim() in [4, 5], "Input must be 4D (spatial) or 5D (volumetric)"
    assert grid.dim() == input.dim(), "Grid must have same dimensionality as input"
    
    is_volumetric = input.dim() == 5
    N, C = input.shape[0], input.shape[1]
    D_in = input.shape[2] if is_volumetric else 0
    H_in, W_in = input.shape[-2], input.shape[-1]
    
    # Output shape
    if is_volumetric:
        N_grid, D_out, H_out, W_out, _ = grid.shape
        output = torch.empty((N, C, D_out, H_out, W_out), device=input.device, dtype=input.dtype)
    else:
        N_grid, H_out, W_out, _ = grid.shape
        output = torch.empty((N, C, H_out, W_out), device=input.device, dtype=input.dtype)
    
    # Strides
    stride_input_n = input.stride(0)
    stride_input_c = input.stride(1)
    stride_input_d = input.stride(2) if is_volumetric else 0
    stride_input_h = input.stride(-2)
    stride_input_w = input.stride(-1)
    
    stride_grid_n = grid.stride(0)
    stride_grid_d = grid.stride(1) if is_volumetric else 0
    stride_grid_h = grid.stride(-3) if is_volumetric else grid.stride(1)
    stride_grid_w = grid.stride(-2) if is_volumetric else grid.stride(2)
    stride_grid_c = grid.stride(-1)
    
    stride_output_n = output.stride(0)
    stride_output_c = output.stride(1)
    stride_output_d = output.stride(2) if is_volumetric else 0
    stride_output_h = output.stride(-2)
    stride_output_w = output.stride(-1)
    
    # Launch kernel
    grid_args = {
        'input_ptr': input,
        'grid_ptr': grid,
        'output_ptr': output,
        'input_N': N,
        'input_C': C,
        'input_D': D_in,
        'input_H': H_in,
        'input_W': W_in,
        'grid_N': N,
        'grid_D': D_out if is_volumetric else 0,
        'grid_H': H_out,
        'grid_W': W_out,
        'grid_C': grid.size(-1),
        'output_N': N,
        'output_C': C,
        'output_D': D_out if is_volumetric else 0,
        'output_H': H_out,
        'output_W': W_out,
        'mode': 0 if mode == 'nearest' else 1,  # Placeholder for mode handling
        'padding_mode': 0,  # Placeholder for padding mode
        'align_corners': align_corners,
        'stride_input_n': stride_input_n,
        'stride_input_c': stride_input_c,
        'stride_input_d': stride_input_d,
        'stride_input_h': stride_input_h,
        'stride_input_w': stride_input_w,
        'stride_grid_n': stride_grid_n,
        'stride_grid_d': stride_grid_d,
        'stride_grid_h': stride_grid_h,
        'stride_grid_w': stride_grid_w,
        'stride_grid_c': stride_grid_c,
        'stride_output_n': stride_output_n,
        'stride_output_c': stride_output_c,
        'stride_output_d': stride_output_d,
        'stride_output_h': stride_output_h,
        'stride_output_w': stride_output_w,
        'BLOCK_SIZE': 32,
    }
    
    if is_volumetric:
        grid_dims = (N, D_out, H_out, W_out)
    else:
        grid_dims = (N, H_out, W_out)
    
    grid_sample_kernel[grid_dims](**grid_args)
    return output
