import torch
import triton
import triton.language as tl

@triton.jit
def affine_grid_kernel(
    theta_ptr,
    grid_ptr,
    H_out,
    W_out,
    align_corners,
    theta_stride_n,
    theta_stride_2,
    theta_stride_3,
    grid_stride_n,
    grid_stride_h,
    grid_stride_w,
    grid_stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(H_out * W_out, BLOCK_SIZE)
    n = pid // num_pid_n
    pid_in_group = pid % num_pid_n
    hw_start = pid_in_group * BLOCK_SIZE
    offsets = hw_start + tl.arange(0, BLOCK_SIZE)
    hw_mask = offsets < H_out * W_out
    h_idx = offsets // W_out
    w_idx = offsets % W_out

    theta_n_ptr = theta_ptr + n * theta_stride_n
    theta_00 = tl.load(theta_n_ptr + 0 * theta_stride_2 + 0 * theta_stride_3)
    theta_01 = tl.load(theta_n_ptr + 0 * theta_stride_2 + 1 * theta_stride_3)
    theta_02 = tl.load(theta_n_ptr + 0 * theta_stride_2 + 2 * theta_stride_3)
    theta_10 = tl.load(theta_n_ptr + 1 * theta_stride_2 + 0 * theta_stride_3)
    theta_11 = tl.load(theta_n_ptr + 1 * theta_stride_2 + 1 * theta_stride_3)
    theta_12 = tl.load(theta_n_ptr + 1 * theta_stride_2 + 2 * theta_stride_3)

    h = tl.where(hw_mask, h_idx, 0)
    w = tl.where(hw_mask, w_idx, 0)

    if align_corners:
        x_norm = (w / (W_out - 1.0)) * 2.0 - 1.0
        y_norm = (h / (H_out - 1.0)) * 2.0 - 1.0
    else:
        x_norm = ((w + 0.5) / W_out) * 2.0 - 1.0
        y_norm = ((h + 0.5) / H_out) * 2.0 - 1.0

    x_prime = theta_00 * x_norm + theta_01 * y_norm + theta_02
    y_prime = theta_10 * x_norm + theta_11 * y_norm + theta_12

    grid_n_ptr = grid_ptr + n * grid_stride_n
    for c in range(2):
        grid_hw_ptr = grid_n_ptr + h_idx * grid_stride_h + w_idx * grid_stride_w + c * grid_stride_c
        value = tl.where(c == 0, x_prime, y_prime)
        tl.store(grid_hw_ptr, value, mask=hw_mask)

@triton.jit
def grid_sample_kernel(
    input_ptr,
    grid_ptr,
    output_ptr,
    H_in,
    W_in,
    H_out,
    W_out,
    mode,
    padding_mode,
    align_corners,
    input_stride_n,
    input_stride_c,
    input_stride_h,
    input_stride_w,
    grid_stride_n,
    grid_stride_h,
    grid_stride_w,
    grid_stride_c,
    output_stride_n,
    output_stride_c,
    output_stride_h,
    output_stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.cdiv(H_out * W_out, BLOCK_SIZE)
    n = pid // num_pid
    pid_in_group = pid % num_pid
    hw_start = pid_in_group * BLOCK_SIZE
    offsets = hw_start + tl.arange(0, BLOCK_SIZE)
    hw_mask = offsets < H_out * W_out
    h_idx = offsets // W_out
    w_idx = offsets % W_out

    grid_n_ptr = grid_ptr + n * grid_stride_n
    grid_h_ptr = grid_n_ptr + h_idx * grid_stride_h
    grid_w_ptr = grid_h_ptr + w_idx * grid_stride_w
    x = tl.load(grid_w_ptr + 0 * grid_stride_c, mask=hw_mask)
    y = tl.load(grid_w_ptr + 1 * grid_stride_c, mask=hw_mask)

    if align_corners:
        ix = (x + 1) * 0.5 * (W_in - 1)
        iy = (y + 1) * 0.5 * (H_in - 1)
    else:
        ix = (x + 1) * 0.5 * W_in - 0.5
        iy = (y + 1) * 0.5 * H_in - 0.5

    ix = tl.minimum(tl.maximum(ix, 0.0), W_in - 1.0)
    iy = tl.minimum(tl.maximum(iy, 0.0), H_in - 1.0)

    ix0 = tl.math.floor(ix)
    iy0 = tl.math.floor(iy)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    wx = ix - ix0
    wy = iy - iy0

    ix0 = tl.where(ix0 < 0, 0, ix0)
    iy0 = tl.where(iy0 < 0, 0, iy0)
    ix1 = tl.where(ix1 >= W_in, W_in - 1, ix1)
    iy1 = tl.where(iy1 >= H_in, H_in - 1, iy1)

    input_n_ptr = input_ptr + n * input_stride_n
    output_n_ptr = output_ptr + n * output_stride_n

    for c in tl.static_range(0, input_stride_c // input_stride_n):
        input_c_ptr = input_n_ptr + c * input_stride_c
        output_c_ptr = output_n_ptr + c * output_stride_c

        iy0_idx = iy0.to(tl.int32)
        ix0_idx = ix0.to(tl.int32)
        iy1_idx = iy1.to(tl.int32)
        ix1_idx = ix1.to(tl.int32)

        v00_ptr = input_c_ptr + iy0_idx * input_stride_h + ix0_idx * input_stride_w
        v00 = tl.load(v00_ptr, mask=hw_mask & (iy0_idx >= 0) & (ix0_idx >= 0), other=0.0)
        v01_ptr = input_c_ptr + iy0_idx * input_stride_h + ix1_idx * input_stride_w
        v01 = tl.load(v01_ptr, mask=hw_mask & (iy0_idx >= 0) & (ix1_idx < W_in), other=0.0)
        v10_ptr = input_c_ptr + iy1_idx * input_stride_h + ix0_idx * input_stride_w
        v10 = tl.load(v10_ptr, mask=hw_mask & (iy1_idx < H_in) & (ix0_idx >= 0), other=0.0)
        v11_ptr = input_c_ptr + iy1_idx * input_stride_h + ix1_idx * input_stride_w
        v11 = tl.load(v11_ptr, mask=hw_mask & (iy1_idx < H_in) & (ix1_idx < W_in), other=0.0)

        w00 = (1 - wx) * (1 - wy)
        w01 = wx * (1 - wy)
        w10 = (1 - wx) * wy
        w11 = wx * wy

        interpolated = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11

        output_hw_ptr = output_c_ptr + h_idx * output_stride_h + w_idx * output_stride_w
        tl.store(output_hw_ptr, interpolated, mask=hw_mask)

def grid_sample_with_affine(
    input: torch.Tensor,
    theta: torch.Tensor,
    size: torch.Size,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = False
) -> torch.Tensor:
    assert input.dim() == 4, "Input must be 4D (N, C, H_in, W_in)"
    assert theta.dim() == 3 and theta.shape[1:] == (2, 3), "Theta must be (N, 2, 3)"
    N, C, H_out, W_out = size
    grid = torch.empty((N, H_out, W_out, 2), dtype=input.dtype, device=input.device)
    H_in, W_in = input.shape[2], input.shape[3]

    def affine_grid_meta(_):
        return (triton.cdiv(N * H_out * W_out, 512),)
    affine_grid_kernel[affine_grid_meta](
        theta, grid, H_out, W_out, align_corners,
        theta.stride(0), theta.stride(1), theta.stride(2),
        grid.stride(0), grid.stride(1), grid.stride(2), grid.stride(3),
        BLOCK_SIZE=512
    )

    output = torch.empty((N, C, H_out, W_out), dtype=input.dtype, device=input.device)

    def grid_sample_meta(_):
        return (triton.cdiv(N * H_out * W_out, 256),)
    grid_sample_kernel[grid_sample_meta](
        input, grid, output, H_in, W_in, H_out, W_out, mode, padding_mode, align_corners,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        grid.stride(0), grid.stride(1), grid.stride(2), grid.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE=256
    )

    return output
