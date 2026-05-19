import torch
import triton
import triton.language as tl


@triton.jit
def _adaptive_avg_pool2d(
    X,
    Y,
    stride_x_n,
    stride_x_c,
    stride_x_h,
    stride_x_w,
    stride_y_n,
    stride_y_c,
    stride_y_h,
    stride_y_w,
    hkernel,
    wkernel,
    upfactor_h,
    upfactor_w,
    height,
    width,
    csize,
    BLOCK_HW: tl.constexpr,
    GROUPS: tl.constexpr,
):
    pidhm = tl.program_id(0)
    pidn = tl.program_id(1)
    pidc = tl.program_id(2)

    # Compute the spatial index of all block elements
    ry = pidhm % upfactor_h
    rx = pidhm // upfactor_h

    # Load the tiles
    y_offset = (
        pidn * stride_y_n
        + pidc * csize * stride_y_c
        + ry * hkernel * stride_y_h
        + rx * wkernel * stride_y_w
    )
    y_tile_ptr = Y + y_offset + tl.arange(0, BLOCK_HW)

    x_offset = (
        pidn * stride_x_n
        + pidc * csize * stride_x_c
        + (ry * hkernel + tl.arange(0, 1)[:, None]) * stride_x_h
        + (rx * wkernel + tl.arange(0, 1)[None, :]) * stride_x_w
    )

    x0_mask = x_offset < (pidn * stride_x_n + (height - hkernel + 1) * stride_x_h)
    x1_mask = x_offset < (pidn * stride_x_n + height * stride_x_h)
    x2_mask = x_offset < (pidn * stride_x_n + (width - wkernel + 1) * stride_x_w)
    x3_mask = x_offset < (pidn * stride_x_n + width * stride_x_w)

    x0 = tl.load(X + x_offset, mask=x0_mask & x2_mask, other=0)
    x1 = tl.load(
        X + x_offset + 1 * stride_x_w,
        mask=(x0_mask | x1_mask) & (x2_mask | x3_mask),
        other=0,
    )
    x2 = tl.load(
        X + x_offset + 1 * stride_x_h * stride_x_w,
        mask=(x0_mask | x2_mask) & (x1_mask | x3_mask),
        other=0,
    )
    x3 = tl.load(
        X + x_offset + 1 * stride_x_h * stride_x_w + 1 * stride_x_w,
        mask=x1_mask & x3_mask,
        other=0,
    )

    y0 = (x0 + x2) * 0.5
    y1 = (x1 + x3) * 0.5
    y2 = (y0 + y1) * 0.5

    # Write back the results
    tl.store(y_tile_ptr, y2)


def adaptive_avg_pool2d(x, output_size):
    has_groups = isinstance(output_size, tuple)

    if not isinstance(output_size, tuple):
        output_size = (output_size, output_size)

    if len(output_size) != 2:
        raise ValueError("output_size must have length 2")

    dim = x.ndim

    if dim not in [4, 3]:
        raise ValueError("Only NCHW and NHWC formats are currently supported")

    if dim == 3:
        x = x.unsqueeze(0)

    groups = x.shape[1]  # C
    out_h = output_size[0]
    out_w = output_size[1]
    in_h = x.shape[2].item()
    in_w = x.shape[3].item()

    hkernel = in_h // out_h
    wkernel = in_w // out_w
    upfactor_h = in_h / out_h
    upfactor_w = in_w / out_w

    y_shape = (groups, out_h, out_w) if has_groups else (out_h, out_w)
    y = torch.empty(y_shape, device=x.device, dtype=x.dtype)

    csize = groups

    grid = lambda META: (int(out_h * out_w), x.shape[0], 1)
    num_warps = 1

    _adaptive_avg_pool2d[grid](
        x,
        y,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        hkernel,
        wkernel,
        upfactor_h,
        upfactor_w,
        in_h,
        in_w,
        csize,
        BLOCK_HW=out_h * out_w,
        num_warps=num_warps,
        GROUPS=groups,
    )

    if dim == 3:
        y = y.squeeze(0)

    return y
