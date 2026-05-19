import torch
import triton
import triton.language as tl
import math

@triton.jit
def _grid_sample_kernel(
    # Pointers to matrices
    input_ptr, grid_ptr, output_ptr,
    # Matrix dimensions
    batch, channels, height_in, width_in,
    height_out, width_out,
    # Strides for the different dimensions
    stride_batch, stride_channel, stride_height, stride_width,
    grid_stride_batch, grid_stride_h, grid_stride_w,
    out_stride_batch, out_stride_channel, out_stride_height, out_stride_width,
    # Options
    mode_idx, padding_mode_idx,
    BLOCK_SIZE: tl.constexpr
):
    # Compute position in output tensor
    pid = tl.program_id(0)
    num_programs = batch * height_out * width_out
    
    # Calculate indices
    batch_idx = pid // (height_out * width_out)
    hw_idx = pid % (height_out * width_out)
    h_out_idx = hw_idx // width_out
    w_out_idx = hw_idx % width_out
    
    # Early exit if out of bounds
    if batch_idx >= batch:
        return
        
    # Load grid coordinates
    grid_offset = batch_idx * grid_stride_batch + h_out_idx * grid_stride_h + w_out_idx * grid_stride_w
    x = tl.load(grid_ptr + grid_offset)
    y = tl.load(grid_ptr + grid_offset + 1)
    
    # Convert normalized coordinates to pixel coordinates
    x = (x + 1) * (width_in - 1) * 0.5
    y = (y + 1) * (height_in - 1) * 0.5
    
    # Calculate base sampling points
    x0 = tl.floor(x)
    y0 = tl.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Clip coordinates based on padding mode
    if padding_mode_idx == 0:  # zeros
        x0 = tl.clip(x0, 0, width_in - 1)
        x1 = tl.clip(x1, 0, width_in - 1)
        y0 = tl.clip(y0, 0, height_in - 1)
        y1 = tl.clip(y1, 0, height_in - 1)
    
    # Calculate weights for bilinear interpolation
    wx = x - x0
    wy = y - y0
    
    # Offset for the batch
    batch_offset = batch_idx * stride_batch
    
    # For each channel
    for c in range(0, channels, BLOCK_SIZE):
        channel_offset = c * stride_channel
        
        # Load values for bilinear interpolation
        v00 = tl.load(input_ptr + batch_offset + channel_offset + y0 * stride_height + x0 * stride_width)
        v01 = tl.load(input_ptr + batch_offset + channel_offset + y0 * stride_height + x1 * stride_width)
        v10 = tl.load(input_ptr + batch_offset + channel_offset + y1 * stride_height + x0 * stride_width)
        v11 = tl.load(input_ptr + batch_offset + channel_offset + y1 * stride_height + x1 * stride_width)
        
        # Bilinear interpolation
        out = (v00 * (1 - wx) * (1 - wy) +
               v01 * wx * (1 - wy) +
               v10 * (1 - wx) * wy +
               v11 * wx * wy)
        
        # Store result
        out_offset = (batch_idx * out_stride_batch + 
                     c * out_stride_channel +
                     h_out_idx * out_stride_height +
                     w_out_idx * out_stride_width)
        tl.store(output_ptr + out_offset, out)

def grid_sample_with_affine(
    input: torch.Tensor,
    theta: torch.Tensor,
    size: torch.Size,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = False
) -> torch.Tensor:
    """
    Apply affine transformation and grid sampling to the input tensor.
    
    Args:
        input: Input tensor of shape (N, C, H_in, W_in)
        theta: Affine transformation matrix of shape (N, 2, 3)
        size: Target output size as (N, C, H_out, W_out)
        mode: Interpolation mode ('bilinear', 'nearest', 'bicubic')
        padding_mode: Padding mode ('zeros', 'border', 'reflection')
        align_corners: If True, aligns corners for transformation consistency
        
    Returns:
        Transformed tensor of shape (N, C, H_out, W_out)
    """
    assert input.dim() == 4, "Input tensor must be 4D (N, C, H, W)"
    assert theta.size(1) == 2 and theta.size(2) == 3, "Theta must be of shape (N, 2, 3)"
    
    # Generate sampling grid
    N, C, H_out, W_out = size
    grid = torch.nn.functional.affine_grid(theta, size, align_corners=align_corners)
    
    # Prepare output tensor
    output = torch.empty(size, device=input.device, dtype=input.dtype)
    
    # Convert mode to index
    mode_idx = {'bilinear': 0, 'nearest': 1, 'bicubic': 2}[mode]
    padding_mode_idx = {'zeros': 0, 'border': 1, 'reflection': 2}[padding_mode]
    
    # Launch kernel
    grid = grid.contiguous()
    input = input.contiguous()
    
    def grid_divide_factor(align_corners):
        return 1.0 if align_corners else 2.0
    
    # Launch triton kernel
    grid_ptr = grid.data_ptr()
    input_ptr = input.data_ptr()
    output_ptr = output.data_ptr()
    
    # Calculate strides
    input_strides = input.stride()
    grid_strides = grid.stride()
    output_strides = output.stride()
    
    # Configure grid
    num_warps = 4
    BLOCK_SIZE = 32
    
    grid = lambda meta: (
        triton.cdiv(N * H_out * W_out, BLOCK_SIZE),
    )
    
    # Launch kernel
    _grid_sample_kernel[grid](
        input_ptr, grid_ptr, output_ptr,
        N, C, input.size(2), input.size(3),
        H_out, W_out,
        input_strides[0], input_strides[1], input_strides[2], input_strides[3],
        grid_strides[0], grid_strides[1], grid_strides[2],
        output_strides[0], output_strides[1], output_strides[2], output_strides[3],
        mode_idx, padding_mode_idx,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return output
