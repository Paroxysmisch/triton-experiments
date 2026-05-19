import triton
import triton.language as tl

@triton.jit
def affine_grid_kernel(
    input_ptr, theta_ptr, output_ptr, N, C, H_in, W_in, H_out, W_out,
    mode, padding_mode, align_corners, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_stride = H_out * W_out
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_stride
    offsets = tl.where(mask, offsets, 0)

    # Compute the grid coordinates
    y = offsets // W_out
    x = offsets % W_out

    # Compute the normalized coordinates
    if align_corners:
        y = (2.0 * y / (H_out - 1)) - 1.0
        x = (2.0 * x / (W_out - 1)) - 1.0
    else:
        y = 2.0 * (y + 0.5) / H_out - 1.0
        x = 2.0 * (x + 0.5) / W_out - 1.0

    # Apply the affine transformation
    theta = tl.load(theta_ptr + pid * 6, mask=mask, other=0.0)
    theta_00 = theta[0]
    theta_01 = theta[1]
    theta_02 = theta[2]
    theta_10 = theta[3]
    theta_11 = theta[4]
    theta_12 = theta[5]

    x_out = theta_00 * x + theta_01 * y + theta_02
    y_out = theta_10 * x + theta_11 * y + theta_12

    # Clamp the coordinates based on padding mode
    if padding_mode == 0:  # 'zeros'
        x_out = tl.where((x_out < -1.0) | (x_out > 1.0), -2.0, x_out)
        y_out = tl.where((y_out < -1.0) | (y_out > 1.0), -2.0, y_out)
    elif padding_mode == 1:  # 'border'
        x_out = tl.where(x_out < -1.0, -1.0, x_out)
        x_out = tl.where(x_out > 1.0, 1.0, x_out)
        y_out = tl.where(y_out < -1.0, -1.0, y_out)
        y_out = tl.where(y_out > 1.0, 1.0, y_out)
    elif padding_mode == 2:  # 'reflection'
        x_out = tl.where(x_out < -1.0, -1.0 - (x_out + 1.0), x_out)
        x_out = tl.where(x_out > 1.0, 1.0 - (x_out - 1.0), x_out)
        y_out = tl.where(y_out < -1.0, -1.0 - (y_out + 1.0), y_out)
        y_out = tl.where(y_out > 1.0, 1.0 - (y_out - 1.0), y_out)

    # Convert to input tensor coordinates
    if align_corners:
        x_out = (x_out + 1.0) * (W_in - 1) / 2.0
        y_out = (y_out + 1.0) * (H_in - 1) / 2.0
    else:
        x_out = (x_out + 1.0) * W_in / 2.0 - 0.5
        y_out = (y_out + 1.0) * H_in / 2.0 - 0.5

    # Interpolation
    x0 = tl.floor(x_out).to(tl.int32)
    y0 = tl.floor(y_out).to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Clamp to input tensor boundaries
    x0 = tl.where(x0 < 0, 0, x0)
    x0 = tl.where(x0 >= W_in, W_in - 1, x0)
    x1 = tl.where(x1 < 0, 0, x1)
    x1 = tl.where(x1 >= W_in, W_in - 1, x1)
    y0 = tl.where(y0 < 0, 0, y0)
    y0 = tl.where(y0 >= H_in, H_in - 1, y0)
    y1 = tl.where(y1 < 0, 0, y1)
    y1 = tl.where(y1 >= H_in, H_in - 1, y1)

    # Load the values from the input tensor
    input_stride = C * H_in * W_in
    input_offset = (pid * input_stride) + (y0 * W_in + x0) * C
    Ia = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    Ib = tl.load(input_ptr + input_offset + C, mask=mask, other=0.0)
    Ic = tl.load(input_ptr + input_offset + C * W_in, mask=mask, other=0.0)
    Id = tl.load(input_ptr + input_offset + C * W_in + C, mask=mask, other=0.0)

    # Bilinear interpolation
    wa = (x1 - x_out) * (y1 - y_out)
    wb = (x1 - x_out) * (y_out - y0)
    wc = (x_out - x0) * (y1 - y_out)
    wd = (x_out - x0) * (y_out - y0)

    output = wa * Ia + wb * Ib + wc * Ic + wd * Id

    # Store the output
    output_offset = (pid * batch_stride + offsets) * C
    tl.store(output_ptr + output_offset, output, mask=mask)

import torch
import triton

def grid_sample_with_affine(input: torch.Tensor, theta: torch.Tensor, size: torch.Size, mode: str = 'bilinear', padding_mode: str = 'zeros', align_corners: bool = False) -> torch.Tensor:
    N, C, H_in, W_in = input.shape
    H_out, W_out = size[2], size[3]

    # Convert padding mode to integer
    padding_mode_map = {'zeros': 0, 'border': 1, 'reflection': 2}
    padding_mode_int = padding_mode_map[padding_mode]

    # Convert mode to integer
    mode_map = {'bilinear': 0, 'nearest': 1, 'bicubic': 2}
    mode_int = mode_map[mode]

    # Allocate output tensor
    output = torch.empty((N, C, H_out, W_out), device=input.device, dtype=input.dtype)

    # Launch the Triton kernel
    grid = (N * H_out * W_out, )
    affine_grid_kernel[grid](
        input, theta, output, N, C, H_in, W_in, H_out, W_out,
        mode_int, padding_mode_int, align_corners, 1024
    )

    return output

# Sample inputs
input = torch.randn(2, 3, 10, 10, device='cuda')
theta = torch.randn(2, 2, 3, device='cuda')
size = (2, 3, 20, 20)

# PyTorch implementation
grid = torch.nn.functional.affine_grid(theta, size, align_corners=False)
output_torch = torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

# Triton implementation
output_triton = grid_sample_with_affine(input, theta, size, mode='bilinear', padding_mode='zeros', align_corners=False)

# Verify the results
print(torch.allclose(output_torch, output_triton, atol=1e-5))
