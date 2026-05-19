import triton
import triton.language as tl

@triton.jit
def affine_grid_kernel(theta_ptr, grid_ptr, N, H_out, W_out, BLOCK_SIZE: tl.constexpr):
    # Pointers for the input theta matrix and output grid
    n_idx = tl.program_id(0)
    
    # Load the affine transformation matrix for the current batch
    theta = tl.load(theta_ptr + n_idx * 6)
    
    # Calculate the coordinates in the output grid
    h_idx = tl.arange(0, BLOCK_SIZE)
    w_idx = tl.arange(0, BLOCK_SIZE)
    
    # Create normalized grid coordinates
    h_normalized = (h_idx / (H_out - 1)) * 2 - 1
    w_normalized = (w_idx / (W_out - 1)) * 2 - 1
    
    # Calculate affine transformation
    x = theta[0] * w_normalized + theta[1] * h_normalized + theta[2]
    y = theta[3] * w_normalized + theta[4] * h_normalized + theta[5]
    
    # Store the result in the output grid
    tl.store(grid_ptr + n_idx * H_out * W_out * 2 + h_idx * W_out + w_idx, x)
    tl.store(grid_ptr + n_idx * H_out * W_out * 2 + h_idx * W_out + w_idx + H_out * W_out, y)

@triton.jit
def grid_sample_kernel(input_ptr, grid_ptr, output_ptr, N, C, H_in, W_in, H_out, W_out, mode, padding_mode, align_corners, BLOCK_SIZE: tl.constexpr):
    # Pointers for input, grid, and output
    n_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    
    # Load grid values
    h_idx = tl.arange(0, BLOCK_SIZE)
    w_idx = tl.arange(0, BLOCK_SIZE)
    
    x = tl.load(grid_ptr + n_idx * H_out * W_out * 2 + h_idx * W_out + w_idx)
    y = tl.load(grid_ptr + n_idx * H_out * W_out * 2 + h_idx * W_out + w_idx + H_out * W_out)
    
    # Implement interpolation based on mode
    # This is a simplified example assuming 'bilinear' mode
    x0 = tl.floor(x).to(tl.int32)
    x1 = x0 + 1
    y0 = tl.floor(y).to(tl.int32)
    y1 = y0 + 1
    
    # Load input tensor values and perform interpolation
    # Handle padding_mode and align_corners as needed
    # For simplicity, assume 'zeros' padding and no align_corners
    i00 = tl.load(input_ptr + n_idx * C * H_in * W_in + c_idx * H_in * W_in + y0 * W_in + x0, mask=(x0 >= 0) & (x0 < W_in) & (y0 >= 0) & (y0 < H_in), other=0)
    i01 = tl.load(input_ptr + n_idx * C * H_in * W_in + c_idx * H_in * W_in + y0 * W_in + x1, mask=(x1 >= 0) & (x1 < W_in) & (y0 >= 0) & (y0 < H_in), other=0)
    i10 = tl.load(input_ptr + n_idx * C * H_in * W_in + c_idx * H_in * W_in + y1 * W_in + x0, mask=(x0 >= 0) & (x0 < W_in) & (y1 >= 0) & (y1 < H_in), other=0)
    i11 = tl.load(input_ptr + n_idx * C * H_in * W_in + c_idx * H_in * W_in + y1 * W_in + x1, mask=(x1 >= 0) & (x1 < W_in) & (y1 >= 0) & (y1 < H_in), other=0)
    
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    
    # Compute output values
    output_value = wa * i00 + wb * i01 + wc * i10 + wd * i11
    
    # Store the result in the output tensor
    tl.store(output_ptr + n_idx * C * H_out * W_out + c_idx * H_out * W_out + h_idx * W_out + w_idx, output_value)

import torch

def grid_sample_with_affine(input: torch.Tensor, theta: torch.Tensor, size: torch.Size, mode: str = 'bilinear', padding_mode: str = 'zeros', align_corners: bool = False) -> torch.Tensor:
    # Extract dimensions
    N, C, H_in, W_in = input.shape
    _, _, H_out, W_out = size
    
    # Allocate memory for the output and grid
    output = torch.empty((N, C, H_out, W_out), device=input.device, dtype=input.dtype)
    grid = torch.empty((N, H_out, W_out, 2), device=input.device, dtype=input.dtype)
    
    # Launch the affine grid kernel
    affine_grid_kernel[(N,)](theta, grid, N, H_out, W_out, BLOCK_SIZE=32)
    
    # Launch the grid sample kernel
    grid_sample_kernel[(N, C)](input, grid, output, N, C, H_in, W_in, H_out, W_out, mode, padding_mode, align_corners, BLOCK_SIZE=32)
    
    return output
