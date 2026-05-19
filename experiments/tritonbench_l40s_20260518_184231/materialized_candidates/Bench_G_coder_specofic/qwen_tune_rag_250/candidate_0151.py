group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M, ngroups)
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[grid](x, out, weight, bias, z, mean, rstd,
                                           x.stride(0), out.stride(0), z.stride(0) if z is not None else 0,
                                           M, group_size, eps,
                                           BLOCK_N=BLOCK_N,
                                           NORM_BEFORE_GATE=norm_before_gate,
                                           IS_RMS_NORM=is_rms_norm,
                                           num_warps=num_warps)
    return out, mean, rstd


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None})
@triton.jit
def _layer_norm_bwd_kernel(
    X,  # pointer to the input
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Y,  # pointer to the output to be recomputed
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    DZ,  # pointer to the other branch
    DX1,  # pointer to the first branch of the second input gradient
    DY1,  # pointer to the first branch of the output gradient
    DW1,  # pointer to the first branch of the weights gradient
    DB1,  # pointer to the first branch of the biases gradient
    DZ1,  # pointer to the first branch of the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_z_row,
    stride_y_row,
    stride_dy_row,
    stride_dx_row,
    stride_dz_row,
    stride_dy1_row,
    stride_dx1_row,
    stride_dz1_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    rows_per_program,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row_block_id = tl.program_id(0)
    group = tl.program_id(1)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row + group * N
    if HAS_Z:
        Z += row_start * stride_z_row + group * N
        DZ += row_start * stride_dz_row + group * N
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row + group * N
    DY += row_start * stride_dy_row + group * N
    DX += row_start * stride_dx_row + group * N
    if HAS_Z and NORM_BEFORE_GATE:
        DZ1 += row_start * stride_dz1_row + group * N
    DX1 += row_start * stride_dx1_row + group * N
    if RECOMPUTE_OUTPUT:
        DY1 += row_start * stride_dy1_row + group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_Z and not NORM_BEFORE_GATE:
        ZG = tl.load(Z + cols, mask=mask, other=0.).to(tl.float32)
        ZG = ZG * tl.sigmoid(ZG)
    else:
        ZG = None
    if RECOMPUTE_OUTPUT:
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row_start + tl.arange(0, rows_per_program), mask=(row_start + tl.arange(0, rows_per_program)) < M).to(tl.float32)
        rstd = tl.load(Rstd + row_start + tl.arange(0, rows_per_program), mask=(row_start + tl.arange(0, rows_per_program)) < M).to(tl.float32)
    # Compute linear transformation and gate
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    if RECOMPUTE_OUTPUT:
        y = (x - mean) * rstd * w if not IS_RMS_NORM else (x * rstd) * w
        tl.store(Y + cols, y, mask=mask)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0).to(tl.float32)
    gw = tl.sum((x - mean) * rstd * dy if not IS_RMS_NORM else dy * rstd, axis=0) if not IS_RMS_NORM else tl.sum(dy * rstd, axis=0)
    if HAS_Z and NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=mask, other=0.).to(tl.float32)
        z1 = tl.load(DZ1 + cols, mask=mask, other=0.).to(tl.float32)
        z = z * tl.sigmoid(z)
        z1 = z1 * tl.sigmoid(z1)
        dz = dy * w * z * z1 * (1 + tl.log(tl.sigmoid(z)) + tl.log(tl.sigmoid(z1)))
        tl.store(DZ + cols, dz, mask=mask)
        dx = dy * w * z * rstd if not IS_RMS_NORM else dy * w * z * rstd
    else:
        dz = None
        dx = dy * w * rstd if not IS_RMS_NORM else dy * w * rstd
    if HAS_Z:
        gw += tl.sum((z if ZG is None else ZG) * dx, axis=0)
    if RECOMPUTE_OUTPUT:
        gb = tl.sum(dy * y if HAS_BIAS else dy, axis=0)
    else:
        gb = tl.sum(dy * (x - mean) * rstd * w if HAS_BIAS else dy * (x - mean) * rstd * w, axis=0) if not IS_RMS_NORM else tl.sum(dy * (x * rstd) * w, axis=0)
    tl.atomic_add(DW + cols, gw, mask=mask)
    tl.atomic_add(DB + cols, gb, mask=mask)
    # Update gradients
    dx = dx.to(X.dtype.element_ty)
    tl.store(DX + cols, dx, mask=mask)
    if HAS_Z:
        dz = dz.to(Z.dtype.element_ty)
        tl.store(DZ + cols, dz, mask=mask)
    if RECOMPUTE_OUTPUT:
        dy = dy.to(Y.dtype.element_ty)
        tl.store(DY1 + cols, dy, mask=mask)
    if HAS_Z and NORM_BEFORE_GATE:
        dz1 = tl.load(DZ1 + cols, mask=mask, other=0.).to(tl.float32)
        dz1 = dz1 * tl.sigmoid(dz1)
        dw1 = tl.sum((z * dz1) * dy, axis=0)
        tl
