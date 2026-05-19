import triton
import triton.language as tl
import math
import torch

@triton.jit
def _grid_sample_kernel_4d(
    inp_ptr, grid_ptr, out_ptr,
    n, c, h, w, out_h, out_w,
    stride_inp_nc, stride_inp_ch, stride_inp_hw,
    stride_grid_nhw, stride_grid_hw,
    stride_out_nch, stride_out_chw,
    align_corners, mode, padding_mode,
    BLOCK: tl.constexpr
):
    # pid will index each output pixel across N * out_H * out_W in parallel
    pid = tl.program_id(0)
    # compute N, out_y, out_x from pid
    N = pid // (out_h * out_w)
    tmp = pid % (out_h * out_w)
    out_y = tmp // out_w
    out_x = tmp % out_w

    # clamp N
    if N >= n:
        return

    # read grid values for each pixel
    # grid is (N, out_H, out_W, 2) for 4D input
    grid_offset = N * stride_grid_nhw + out_y * stride_grid_hw + out_x * 2
    gx = tl.load(grid_ptr + grid_offset)
    gy = tl.load(grid_ptr + grid_offset + 1)

    # handle NaN in grid by interpreting as -1
    if tl.isnan(gx):
        gx = -1.0
    if tl.isnan(gy):
        gy = -1.0

    # if align_corners is True, scale coords to [-1, 1] inclusive corners
    # else scale to half-pixel offset
    if align_corners:
        # from [-1,1] range to [0, w-1] or [0, h-1]
        ix = 0.5 * (gx + 1.0) * (w - 1)
        iy = 0.5 * (gy + 1.0) * (h - 1)
    else:
        ix = 0.5 * ((gx + 1.0) * w - 1.0)
        iy = 0.5 * ((gy + 1.0) * h - 1.0)

    # process out-of-bounds based on padding_mode
    def pad_coord(coord, size):
        if padding_mode == 0:  # 'zeros'
            return -1.0 if (coord < 0 or coord > size - 1) else coord
        elif padding_mode == 1:  # 'border'
            return 0.0 if coord < 0 else (size - 1.0 if coord > size - 1 else coord)
        else:  # 'reflection'
            # reflect coords (simple mod-based reflection)
            if coord < 0:
                return -coord if size > 1 else 0
            elif coord > size - 1:
                return 2*(size - 1) - coord if size > 1 else 0
            else:
                return coord

    # map mode strings to integer for kernel
    # 0 -> nearest, 1 -> bilinear
    if mode == 0:  # nearest
        ix = pad_coord(ix, w)
        iy = pad_coord(iy, h)
        # if out of range in zeros mode, keep -1
        if ix < 0 or iy < 0:
            for ch in range(BLOCK):
                if ch < c:
                    tl.store(
                        out_ptr + N * stride_out_nch
                        + ch * stride_out_chw
                        + out_y * out_w
                        + out_x,
                        0.0
                    )
            return
        # clamp after reflection/border
        ix_int = int(tl.floor(ix + 0.5))
        iy_int = int(tl.floor(iy + 0.5))
        for ch in range(BLOCK):
            if ch < c:
                val = tl.load(
                    inp_ptr + N * stride_inp_nc
                    + ch * stride_inp_ch
                    + iy_int * w
                    + ix_int
                )
                tl.store(
                    out_ptr + N * stride_out_nch
                    + ch * stride_out_chw
                    + out_y * out_w
                    + out_x,
                    val
                )
        return

    # if mode == bilinear
    # compute low/high and interpolation weights
    x0 = tl.floor(ix)
    y0 = tl.floor(iy)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = ix - x0
    wy = iy - y0

    x0 = pad_coord(x0, w)
    x1 = pad_coord(x1, w)
    y0 = pad_coord(y0, h)
    y1 = pad_coord(y1, h)
    # if zeros mode yields negative => out of range, skip load

    for ch in range(BLOCK):
        if ch < c:
            # corners
            v00 = 0.0
            v01 = 0.0
            v10 = 0.0
            v11 = 0.0
            if x0 >= 0 and y0 >= 0:
                v00 = tl.load(
                    inp_ptr + N * stride_inp_nc + ch * stride_inp_ch + int(y0) * w + int(x0)
                )
            if x1 >= 0 and y0 >= 0:
                v01 = tl.load(
                    inp_ptr + N * stride_inp_nc + ch * stride_inp_ch + int(y0) * w + int(x1)
                )
            if x0 >= 0 and y1 >= 0:
                v10 = tl.load(
                    inp_ptr + N * stride_inp_nc + ch * stride_inp_ch + int(y1) * w + int(x0)
                )
            if x1 >= 0 and y1 >= 0:
                v11 = tl.load(
                    inp_ptr + N * stride_inp_nc + ch * stride_inp_ch + int(y1) * w + int(x1)
                )

            val = (v00 * (1 - wx) * (1 - wy) +
                   v01 * wx * (1 - wy) +
                   v10 * (1 - wx) * wy +
                   v11 * wx * wy)

            tl.store(
                out_ptr + N * stride_out_nch
                + ch * stride_out_chw
                + out_y * out_w
                + out_x,
                val
            )

def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    """
    Triton wrapper for grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False).
    Supports 4-D input (N, C, H, W) with a grid of shape (N, out_H, out_W, 2).
    Interpolation can be 'nearest' or 'bilinear'. Values outside [-1, 1] range
    are handled via padding_mode ('zeros', 'border', 'reflection').
    If align_corners=True, corners map to the extreme image borders.
    NaN in grid is treated as -1.
    This wrapper dispatches a single Triton kernel to produce the result.
    """
    if mode not in ['nearest', 'bilinear']:
        raise ValueError("Only 'nearest' or 'bilinear' modes implemented in this example.")
    int_mode = 0 if mode == 'nearest' else 1

    if padding_mode not in ['zeros', 'border', 'reflection']:
        raise ValueError("padding_mode must be 'zeros', 'border', or 'reflection'")
    pad_map = {'zeros': 0, 'border': 1, 'reflection': 2}
    int_pad_mode = pad_map[padding_mode]

    # handle an example 4-D input
    if input.ndim != 4:
        raise NotImplementedError("Only 4-D input (N, C, H, W) is implemented in this example.")

    N, C, H, W = input.shape
    out_H, out_W = grid.shape[1], grid.shape[2]

    # create output
    out = torch.empty((N, C, out_H, out_W), dtype=input.dtype, device=input.device)

    # strides for input
    stride_inp_nc = C * H * W
    stride_inp_ch = H * W
    stride_inp_hw = W  # not separately used above, can keep for clarity

    # strides for grid (N, out_H, out_W, 2)
    # we read as (N * out_H * out_W * 2), so:
    stride_grid_nhw = out_H * out_W * 2
    stride_grid_hw = out_W * 2

    # strides for out
    stride_out_nch = C * out_H * out_W
    stride_out_chw = out_H * out_W

    # convert booleans to int for kernel
    align_corners_int = 1 if align_corners else 0

    # launch kernel
    # Each program handles exactly one (N, out_H, out_W) pixel.
    num_programs = N * out_H * out_W
    _grid_sample_kernel_4d[grid=torch.Size([num_programs])](
        inp_ptr=input.data_ptr(),
        grid_ptr=grid.data_ptr(),
        out_ptr=out.data_ptr(),
        n=N,
        c=C,
        h=H,
        w=W,
        out_h=out_H,
        out_w=out_W,
        stride_inp_nc=stride_inp_nc,
        stride_inp_ch=stride_inp_ch,
        stride_inp_hw=stride_inp_hw,
        stride_grid_nhw=stride_grid_nhw,
        stride_grid_hw=stride_grid_hw,
        stride_out_nch=stride_out_nch,
        stride_out_chw=stride_out_chw,
        align_corners=align_corners_int,
        mode=int_mode,
        padding_mode=int_pad_mode,
        BLOCK=C  # we can process channels in a loop for simplicity
    )

    return out
