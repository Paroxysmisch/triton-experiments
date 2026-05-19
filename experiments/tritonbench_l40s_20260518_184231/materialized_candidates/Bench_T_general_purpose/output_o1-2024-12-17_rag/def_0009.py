import torch
import triton
import triton.language as tl

@triton.jit
def _grid_sample_kernel(
    input_ptr,        # float*    (N, C, IH, IW) flattened
    grid_ptr,         # float*    (N, OH, OW, 2) flattened
    output_ptr,       # float*    (N, C, OH, OW) flattened
    N,                # int
    C,                # int
    IH,               # int
    IW,               # int
    OH,               # int
    OW,               # int
    mode,             # int: 0=nearest, 1=bilinear
    padding_mode,     # int: 0=zeros, 1=border, 2=reflection
    align_corners,    # int: 0=False, 1=True
    n_elements,       # int total threads
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Convert 1D index into 4D index [N, OH, OW]
    # offset = n * (OH * OW) + oh * (OW) + ow
    n = offsets // (OH * OW)
    tmp = offsets % (OH * OW)
    oh = tmp // OW
    ow = tmp % OW

    # Load normalized grid coords
    # grid_ptr shape: [N, OH, OW, 2], flattened
    # grid_idx = n*(OH*OW*2) + oh*(OW*2) + ow*(2)
    grid_idx = n * (OH * OW * 2) + oh * (OW * 2) + ow * 2
    gx = tl.load(grid_ptr + grid_idx + 0, mask=mask)
    gy = tl.load(grid_ptr + grid_idx + 1, mask=mask)

    # Replace NaN with -1
    is_nan_x = tl.isnan(gx)
    is_nan_y = tl.isnan(gy)
    gx = tl.where(is_nan_x, -1.0, gx)
    gy = tl.where(is_nan_y, -1.0, gy)

    # align_corners logic
    align = align_corners == 1
    if align:
        ix = (gx + 1) / 2 * (IW - 1)
        iy = (gy + 1) / 2 * (IH - 1)
    else:
        ix = ((gx + 1) * 0.5) * IW - 0.5
        iy = ((gy + 1) * 0.5) * IH - 0.5

    # Handle padding_mode
    pm = padding_mode
    # 0=zeros, 1=border, 2=reflection
    # For brevity only zeros/border are partially shown
    def clamp_coords(coord, size):
        return tl.max(0., tl.min(coord, float(size - 1)))

    if pm == 0:
        # zeros: out of bound -> ignore
        pass
    elif pm == 1:
        # border: clamp
        ix = clamp_coords(ix, IW)
        iy = clamp_coords(iy, IH)
    # reflection could be handled similarly if needed

    # mode logic
    md = mode
    # 0=nearest, 1=bilinear
    for c_ in range(C):
        out_val = 0.0
        if md == 0:
            # Nearest
            nx = tl.round(ix)
            ny = tl.round(iy)
            # check bounds if padding_mode=zeros
            in_bounds = (nx >= 0) & (nx < IW) & (ny >= 0) & (ny < IH)
            base_idx = n * (C * IH * IW) + c_ * (IH * IW)
            sample_idx = base_idx + ny * IW + nx
            val = 0.0
            if pm == 0:
                val = tl.where(in_bounds, tl.load(input_ptr + sample_idx, mask=in_bounds), 0.0)
            else:
                # border or reflection already clamped
                sample_idx = tl.max(0, tl.min(sample_idx, (N*C*IH*IW - 1)))
                val = tl.load(input_ptr + sample_idx, mask=mask)
            out_val = val
        else:
            # Bilinear
            x0 = tl.floor(ix)
            x1 = x0 + 1
            y0 = tl.floor(iy)
            y1 = y0 + 1

            # fractional distances
            wx1 = ix - x0
            wx0 = 1.0 - wx1
            wy1 = iy - y0
            wy0 = 1.0 - wy1

            base_idx = n * (C * IH * IW) + c_ * (IH * IW)

            def load_bilinear(_x, _y, _wx, _wy):
                # handle out of bound -> 0 if zeros
                in_bounds = (_x >= 0) & (_x < IW) & (_y >= 0) & (_y < IH)
                idx_ = base_idx + _y * IW + _x
                val_ = tl.where(
                    in_bounds,
                    tl.load(input_ptr + idx_, mask=in_bounds),
                    0.0
                )
                return val_ * _wx * _wy

            val00 = load_bilinear(x0, y0, wx0, wy0)
            val01 = load_bilinear(x0, y1, wx0, wy1)
            val10 = load_bilinear(x1, y0, wx1, wy0)
            val11 = load_bilinear(x1, y1, wx1, wy1)
            out_val = val00 + val01 + val10 + val11

        # Store result
        out_idx = n * (C * OH * OW) + c_ * (OH * OW) + oh * OW + ow
        tl.store(output_ptr + out_idx, out_val, mask=mask)

def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    # Determine mode int
    mode_map = {'nearest': 0, 'bilinear': 1, 'bicubic': 2}
    mode_int = mode_map.get(mode, 1)

    # Determine padding_mode int
    pm_map = {'zeros': 0, 'border': 1, 'reflection': 2}
    pm_int = pm_map.get(padding_mode, 0)

    align_i = 1 if align_corners else 0

    # Input shape can be 4D: (N, C, H, W)
    # Grid shape can be (N, OH, OW, 2)
    # Output shape: (N, C, OH, OW)
    assert input.dim() in (4, 5), "Only spatial (4D) or volumetric (5D) input supported"
    if input.dim() == 4:
        N, C, IH, IW = input.shape
        # For 2D, grid shape expected: [N, OH, OW, 2]
        assert grid.dim() == 4 and grid.shape[-1] == 2, "Grid must have shape (N, OH, OW, 2)"
        assert grid.shape[0] == N, "Grid batch size must match input"
        OH, OW = grid.shape[1], grid.shape[2]
        output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    else:
        # 5D volumetric case (N, C, D, H, W) - skipping full detail example
        raise NotImplementedError("Volumetric 5D input not implemented in this example.")

    # Flatten
    input_flat = input.contiguous().view(-1)
    grid_flat = grid.contiguous().view(-1)
    output_flat = output.contiguous().view(-1)

    # Total elements to compute in the output is N*OH*OW
    n_elements = N * OH * OW
    grid_fn = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _grid_sample_kernel[grid_fn](
        input_flat, grid_flat, output_flat,
        N, C, IH, IW, OH, OW,
        mode_int, pm_int, align_i,
        n_elements,
        BLOCK_SIZE=1024
    )
    return output
