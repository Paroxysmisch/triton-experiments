import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def grid_sample_2d_kernel(
    input_ptr,
    grid_ptr,
    output_ptr,
    input_n,
    input_c,
    input_h,
    input_w,
    output_n,
    output_c,
    output_h,
    output_w,
    stride_input_n,
    stride_input_c,
    stride_input_h,
    stride_input_w,
    stride_grid_n,
    stride_grid_h,
    stride_grid_w,
    stride_grid_d,
    stride_output_n,
    stride_output_c,
    stride_output_h,
    stride_output_w,
    mode: tl.constexpr,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= output_n * output_c * output_h * output_w:
        return

    # Calculate output indices
    n = pid // (output_c * output_h * output_w)
    pid_remaining = pid % (output_c * output_h * output_w)
    c = pid_remaining // (output_h * output_w)
    pid_remaining = pid_remaining % (output_h * output_w)
    h_out = pid_remaining // output_w
    w_out = pid_remaining % output_w

    # Load grid values
    grid_offset = n * stride_grid_n + h_out * stride_grid_h + w_out * stride_grid_w
    x = tl.load(grid_ptr + grid_offset + 0 * stride_grid_d)
    y = tl.load(grid_ptr + grid_offset + 1 * stride_grid_d)

    # Handle NaN
    x = tl.where(tl.math.isnan(x), -1.0, x)
    y = tl.where(tl.math.isnan(y), -1.0, y)

    # Convert grid to input coordinates
    if align_corners:
        ix = (x + 1.0) * (input_w - 1.0) / 2.0
        iy = (y + 1.0) * (input_h - 1.0) / 2.0
    else:
        ix = (x + 1.0) * input_w / 2.0 - 0.5
        iy = (y + 1.0) * input_h / 2.0 - 0.5

    # Clamp function for border padding
    def clamp(v, size):
        return tl.math.max(0.0, tl.math.min(v, size - 1.0))

    # Reflection function
    def reflect(v, size):
        v = tl.math.abs(v)
        period = size * 2.0
        cycles = tl.math.floor(v / period)
        v = v - cycles * period
        v = tl.where(v >= size, period - v, v)
        return clamp(v, size)

    # Get input value with padding handling
    def get_value(ix_val, iy_val):
        if padding_mode == 'zeros':
            within_x = (ix_val >= 0) & (ix_val < input_w)
            within_y = (iy_val >= 0) & (iy_val < input_h)
            if within_x & within_y:
                ix_idx = tl.math.floor(ix_val)
                iy_idx = tl.math.floor(iy_val)
                offset = n * stride_input_n + c * stride_input_c + iy_idx * stride_input_h + ix_idx * stride_input_w
                return tl.load(input_ptr + offset)
            else:
                return 0.0
        elif padding_mode == 'border':
            ix_clamp = clamp(ix_val, input_w)
            iy_clamp = clamp(iy_val, input_h)
            ix_idx = tl.math.floor(ix_clamp)
            iy_idx = tl.math.floor(iy_clamp)
            offset = n * stride_input_n + c * stride_input_c + iy_idx * stride_input_h + ix_idx * stride_input_w
            return tl.load(input_ptr + offset)
        elif padding_mode == 'reflection':
            ix_reflect = reflect(ix_val, input_w)
            iy_reflect = reflect(iy_val, input_h)
            ix_idx = tl.math.floor(ix_reflect)
            iy_idx = tl.math.floor(iy_reflect)
            offset = n * stride_input_n + c * stride_input_c + iy_idx * stride_input_h + ix_idx * stride_input_w
            return tl.load(input_ptr + offset)
        else:
            return 0.0

    if mode == 'nearest':
        ix_nearest = tl.math.round(ix)
        iy_nearest = tl.math.round(iy)
        result = get_value(ix_nearest, iy_nearest)
    elif mode == 'bilinear':
        ix_floor = tl.math.floor(ix)
        iy_floor = tl.math.floor(iy)
        fx = ix - ix_floor
        fy = iy - iy_floor

        v00 = get_value(ix_floor, iy_floor)
        v01 = get_value(ix_floor, iy_floor + 1)
        v10 = get_value(ix_floor + 1, iy_floor)
        v11 = get_value(ix_floor + 1, iy_floor + 1)

        w00 = (1 - fx) * (1 - fy)
        w01 = (1 - fx) * fy
        w10 = fx * (1 - fy)
        w11 = fx * fy

        result = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
    else:
        result = 0.0

    # Store output
    output_offset = n * stride_output_n + c * stride_output_c + h_out * stride_output_h + w_out * stride_output_w
    tl.store(output_ptr + output_offset, result)

def grid_sample(input: Tensor, grid: Tensor, mode: str = 'bilinear', padding_mode: str = 'zeros', align_corners: bool = False) -> Tensor:
    assert input.dim() in [4, 5], "Input must be 4D or 5D"
    assert grid.dim() == input.dim(), "Grid must have same dimensionality as input"
    assert grid.size(-1) == (3 if input.dim() == 5 else 2), "Grid last dim must match spatial dims"

    N, C = input.shape[0], input.shape[1]
    spatial_dims = grid.shape[1:-1]
    output_shape = (N, C, *spatial_dims)
    output = torch.empty(output_shape, device=input.device, dtype=input.dtype)

    if input.dim() == 4:
        input_h, input_w = input.shape[2], input.shape[3]
        stride_input = input.stride()
        stride_grid = grid.stride()
        stride_output = output.stride()

        BLOCK_SIZE = 128
        num_elements = output.numel()
        grid_args = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
        grid_sample_2d_kernel[grid_args](
            input, grid, output,
            input.size(0), input.size(1), input_h, input_w,
            output.size(0), output.size(1), output.size(2), output.size(3),
            stride_input[0], stride_input[1], stride_input[2], stride_input[3],
            stride_grid[0], stride_grid[1], stride_grid[2], stride_grid[3],
            stride_output[0], stride_output[1], stride_output[2], stride_output[3],
            mode, padding_mode, align_corners, BLOCK_SIZE
        )
    else:
        raise NotImplementedError("Volumetric grid_sample (5D) not implemented")

    return output
