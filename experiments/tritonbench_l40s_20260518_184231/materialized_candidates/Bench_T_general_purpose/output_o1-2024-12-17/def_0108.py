import torch
import triton
import triton.language as tl

# --------------------------------------------
# Kernel: 2D Grid Sampling with optional bilinear/nearest/bicubic interpolation
# --------------------------------------------
@triton.jit
def _grid_sample_2d_kernel(
    input_ptr,  # float32[N, C, H_in, W_in]
    grid_ptr,   # float32[N, H_out, W_out, 2], normalized coords in [-1, 1]
    output_ptr, # float32[N, C, H_out, W_out]
    n_px: tl.constexpr,
    c_px: tl.constexpr,
    h_in: tl.constexpr,
    w_in: tl.constexpr,
    h_out: tl.constexpr,
    w_out: tl.constexpr,
    mode_code: tl.constexpr,         # 0=bilinear, 1=nearest, 2=bicubic
    padding_mode_code: tl.constexpr, # 0=zeros, 1=border, 2=reflection
):
    pid = tl.program_id(0)
    block_size = tl.launch.grid(0)
    idx = pid * block_size + tl.arange(0, block_size)
    mask = idx < (n_px * h_out * w_out)

    # Decompose idx into (n, y_out, x_out)
    n = idx // (h_out * w_out)
    yx_idx = idx % (h_out * w_out)
    y_out = yx_idx // w_out
    x_out = yx_idx % w_out

    # Load normalized coords from grid
    # shape of grid: (N, H_out, W_out, 2)
    # offset for 4D index: offset = ((n * H_out + y_out) * W_out + x_out) * 2
    offset_grid = (n * h_out * w_out + y_out * w_out + x_out) * 2
    norm_x = tl.load(grid_ptr + offset_grid, mask=mask, other=0.0)
    norm_y = tl.load(grid_ptr + offset_grid + 1, mask=mask, other=0.0)

    # Convert normalized [-1,1] -> (0..W_in-1), (0..H_in-1)
    in_x_f = (norm_x + 1) * 0.5 * (w_in - 1)
    in_y_f = (norm_y + 1) * 0.5 * (h_in - 1)

    # Handle padding
    # If padding_mode == zeros, out-of-bounds -> 0
    # If padding_mode == border, clamp to boundary
    # If padding_mode == reflection, reflect coords
    # We'll inline a small function for each:
    def clamp_coords(coord, size):
        return tl.maximum(0., tl.minimum(coord, size - 1))

    def reflect_coords(coord, size):
        # reflect within [0..size-1]
        # compute reflection
        # keep reflecting until in range
        # simplified approach
        mul = tl.floor(coord / (size - 1))
        flip = tl.abs(mul % 2)
        offset = coord % (size - 1)
        reflect_val = tl.where(flip > 1e-5, (size - 1) - offset, offset)
        return reflect_val

    def handle_padding(xf, yf):
        if padding_mode_code == 0:  # zeros
            # we rely on checking OOB after interpolation
            return xf, yf
        elif padding_mode_code == 1:  # border
            return clamp_coords(xf, w_in), clamp_coords(yf, h_in)
        elif padding_mode_code == 2:  # reflection
            return reflect_coords(xf, w_in), reflect_coords(yf, h_in)
        return xf, yf

    in_x_f, in_y_f = handle_padding(in_x_f, in_y_f)

    # Load from input depending on interpolation mode
    # offset for reading input: offset_in = ((n * C + c) * H_in + y_in) * W_in + x_in
    # We'll handle for each c.

    def load_input(n_, c_, y_in_, x_in_):
        # Check OOB for zeros padding
        if padding_mode_code == 0:
            oob_mask = (x_in_ < 0) | (x_in_ >= w_in) | (y_in_ < 0) | (y_in_ >= h_in)
            val = tl.where(oob_mask, 0.0,
                           tl.load(input_ptr + ((n_ * c_px + c_) * h_in + y_in_) * w_in + x_in_))
        else:
            # after handle_padding, in_x_f, in_y_f are in-bounds for border or reflection
            # clamp for final indexing
            x_in_ = tl.maximum(0, tl.minimum(x_in_, w_in - 1))
            y_in_ = tl.maximum(0, tl.minimum(y_in_, h_in - 1))
            val = tl.load(input_ptr + ((n_ * c_px + c_) * h_in + y_in_) * w_in + x_in_)
        return val

    # We'll define bilinear interpolation
    def bilinear(n_, c_, y_f, x_f):
        y0 = tl.floor(y_f)
        x0 = tl.floor(x_f)
        y1 = y0 + 1
        x1 = x0 + 1
        wy1 = y_f - y0
        wx1 = x_f - x0
        wy0 = 1.0 - wy1
        wx0 = 1.0 - wx1

        v00 = load_input(n_, c_, tl.cast(y0, tl.int32), tl.cast(x0, tl.int32))
        v01 = load_input(n_, c_, tl.cast(y0, tl.int32), tl.cast(x1, tl.int32))
        v10 = load_input(n_, c_, tl.cast(y1, tl.int32), tl.cast(x0, tl.int32))
        v11 = load_input(n_, c_, tl.cast(y1, tl.int32), tl.cast(x1, tl.int32))

        return (v00 * wx0 * wy0 +
                v01 * wx1 * wy0 +
                v10 * wx0 * wy1 +
                v11 * wx1 * wy1)

    # Nearest interpolation
    def nearest(n_, c_, y_f, x_f):
        ny = tl.round(y_f)
        nx = tl.round(x_f)
        return load_input(n_, c_, tl.cast(ny, tl.int32), tl.cast(nx, tl.int32))

    # Bicubic is more complex. Here we place a simple placeholder or nearest approximation.
    # True cubic interpolation would require 4x4 neighborhood. 
    # This is a placeholder to show the mode selection mechanism.
    # For real usage, implement the cubic formula as needed.
    def bicubic(n_, c_, y_f, x_f):
        return nearest(n_, c_, y_f, x_f)  # placeholder

    for c in range(c_px):
        val = 0.0
        if mode_code == 0:
            val = bilinear(n, c, in_y_f, in_x_f)
        elif mode_code == 1:
            val = nearest(n, c, in_y_f, in_x_f)
        else:
            val = bicubic(n, c, in_y_f, in_x_f)

        offset_out = ((n * c_px + c) * h_out + y_out) * w_out + x_out
        tl.store(output_ptr + offset_out, val, mask=mask)


# --------------------------------------------
# Python helper for generating the affine grid
# --------------------------------------------
def _affine_grid_2d(theta: torch.Tensor,
                    size: torch.Size,
                    align_corners: bool) -> torch.Tensor:
    # size is (N, C, H_out, W_out)
    N, _, H_out, W_out = size
    # Create normalized meshgrid
    if align_corners and H_out > 1 and W_out > 1:
        xs = torch.linspace(-1, 1, W_out, device=theta.device)
        ys = torch.linspace(-1, 1, H_out, device=theta.device)
    else:
        xs = torch.linspace(-1, 1, W_out, device=theta.device, dtype=theta.dtype) * (W_out / (W_out - 1) if W_out > 1 else 1)
        ys = torch.linspace(-1, 1, H_out, device=theta.device,
