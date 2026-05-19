import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton kernel for grid sampling
@triton.jit
def grid_sample_kernel(
    input_ptr,  # Pointer to input tensor
    grid_ptr,   # Pointer to grid tensor
    output_ptr, # Pointer to output tensor
    n_elements, # Number of elements to process
    H_out,      # Height of the output tensor
    W_out,      # Width of the output tensor
    H_in,       # Height of the input tensor
    W_in,       # Width of the input tensor
    mode: tl.constexpr,  # Interpolation mode: 0 for 'bilinear', 1 for 'nearest', 2 for 'bicubic'
    padding_mode: tl.constexpr,  # Padding mode: 0 for 'zeros', 1 for 'border', 2 for 'reflection'
    align_corners: tl.constexpr,  # Align corners
    BLOCK_SIZE: tl.constexpr  # Block size for processing
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load grid values
    grid_x = tl.load(grid_ptr + offsets * 2, mask=mask, other=0.0)
    grid_y = tl.load(grid_ptr + offsets * 2 + 1, mask=mask, other=0.0)

    # Normalize grid values to [-1, 1]
    if align_corners:
        grid_x = (grid_x + 1) * (W_in - 1) / 2
        grid_y = (grid_y + 1) * (H_in - 1) / 2
    else:
        grid_x = (grid_x + 1) * (W_in - 1) / 2
        grid_y = (grid_y + 1) * (H_in - 1) / 2

    # Apply padding mode
    if padding_mode == 0:  # 'zeros'
        grid_x = tl.where((grid_x < 0) | (grid_x > W_in - 1), 0.0, grid_x)
        grid_y = tl.where((grid_y < 0) | (grid_y > H_in - 1), 0.0, grid_y)
    elif padding_mode == 1:  # 'border'
        grid_x = tl.where(grid_x < 0, 0.0, grid_x)
        grid_x = tl.where(grid_x > W_in - 1, W_in - 1, grid_x)
        grid_y = tl.where(grid_y < 0, 0.0, grid_y)
        grid_y = tl.where(grid_y > H_in - 1, H_in - 1, grid_y)
    elif padding_mode == 2:  # 'reflection'
        grid_x = tl.where(grid_x < 0, -grid_x, grid_x)
        grid_x = tl.where(grid_x > W_in - 1, 2 * (W_in - 1) - grid_x, grid_x)
        grid_y = tl.where(grid_y < 0, -grid_y, grid_y)
        grid_y = tl.where(grid_y > H_in - 1, 2 * (H_in - 1) - grid_y, grid_y)

    # Convert to integer indices
    grid_x = grid_x.to(tl.int32)
    grid_y = grid_y.to(tl.int32)

    # Load input values
    input_values = tl.load(input_ptr + grid_y * W_in + grid_x, mask=mask, other=0.0)

    # Interpolation
    if mode == 0:  # 'bilinear'
        # Bilinear interpolation
        x0 = grid_x.to(tl.float32)
        y0 = grid_y.to(tl.float32)
        x1 = tl.where(grid_x < W_in - 1, grid_x + 1, grid_x)
        y1 = tl.where(grid_y < H_in - 1, grid_y + 1, grid_y)
        x1 = x1.to(tl.float32)
        y1 = y1.to(tl.float32)

        Ia = tl.load(input_ptr + y0 * W_in + x0, mask=mask, other=0.0)
        Ib = tl.load(input_ptr + y1 * W_in + x0, mask=mask, other=0.0)
        Ic = tl.load(input_ptr + y0 * W_in + x1, mask=mask, other=0.0)
        Id = tl.load(input_ptr + y1 * W_in + x1, mask=mask, other=0.0)

        wa = (x1 - grid_x) * (y1 - grid_y)
        wb = (x1 - grid_x) * (grid_y - y0)
        wc = (grid_x - x0) * (y1 - grid_y)
        wd = (grid_x - x0) * (grid_y - y0)

        output_values = wa * Ia + wb * Ib + wc * Ic + wd * Id
    elif mode == 1:  # 'nearest'
        output_values = input_values
    elif mode == 2:  # 'bicubic'
        # Bicubic interpolation (simplified for brevity)
        output_values = input_values  # Placeholder for bicubic interpolation

    # Store the result
    tl.store(output_ptr + offsets, output_values, mask=mask)

# Wrapper function
def grid_sample_with_affine(input: torch.Tensor, theta: torch.Tensor, size: torch.Size, mode: str = 'bilinear', padding_mode: str = 'zeros', align_corners: bool = False) -> torch.Tensor:
    assert input.dim() == 4, "Input tensor must be 4D (N, C, H_in, W_in)"
    assert theta.dim() == 3 and theta.shape[1:] == (2, 3), "Affine matrix must be of shape (N, 2, 3)"
    assert mode in ['bilinear', 'nearest', 'bicubic'], "Mode must be one of 'bilinear', 'nearest', 'bicubic'"
    assert padding_mode in ['zeros', 'border', 'reflection'], "Padding mode must be one of 'zeros', 'border', 'reflection'"

    N, C, H_in, W_in = input.shape
    N, H_out, W_out = size[0], size[2], size[3]

    # Generate the grid
    grid = F.affine_grid(theta, size, align_corners=align_corners)

    # Flatten the grid and input for Triton kernel
    grid = grid.view(N, H_out * W_out, 2).contiguous()
    input = input.view(N, C, H_in * W_in).contiguous()

    # Output tensor
    output = torch.empty((N, C, H_out, W_out), dtype=input.dtype, device=input.device)

    # Define the grid size for the kernel launch
    n_elements = N * H_out * W_out
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch the Triton kernel
    mode_map = {'bilinear': 0, 'nearest': 1, 'bicubic': 2}
    padding_mode_map = {'zeros': 0, 'border': 1, 'reflection': 2}
    grid_sample_kernel[grid](
        input, grid, output.view(N, H_out * W_out, C).contiguous(), n_elements, H_out, W_out, H_in, W_in,
        mode_map[mode], padding_mode_map[padding_mode], align_corners, BLOCK_SIZE=1024
    )

    return output

# Example usage
if __name__ == "__main__":
    N, C, H_in, W_in = 2, 3, 64, 64
    H_out, W_out = 32, 32
    input = torch.randn(N, C, H_in, W_in, device='cuda')
    theta = torch.randn(N, 2, 3, device='cuda')
    size = torch.Size([N, C, H_out, W_out])

    output = grid_sample_with_affine(input, theta, size
