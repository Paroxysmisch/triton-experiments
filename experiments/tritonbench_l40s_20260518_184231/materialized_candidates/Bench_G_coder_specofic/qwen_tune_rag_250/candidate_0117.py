,
        coord_grad_output: torch.Tensor | None = None,
        block_size: int = 64,
        col_offset: int = 0,
    ) -> torch.Tensor:
        (coords,) = ctx.saved_tensors
        if not isinstance(coord_grad_output, torch.Tensor):
            coord_grad_output = torch.zeros_like(coords)
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        third_order_bwd[num_blocks,](
            coords,
            coord_grad_output,
            sph_grad_tensor,
            block_size,
            coords.numel(),
            sph_grad_tensor.numel(),
            col_offset,
            sph_grad_tensor.stride(-2),
        )
        return coord_grad_output

@triton.jit
def third_order_fwd(
    coord_ptr: tl.tensor,
    output_ptr: tl.tensor,
    block_size: tl.constexpr,
    coord_numel: tl.constexpr,
    output_numel: tl.constexpr,
    col_offset: tl.constexpr,
    output_stride: tl.constexpr,
):
    coord_stride = 3
    block_id = tl.program_id(0)
    coord_striding = tl.arange(0, block_size) * coord_stride
    coord_row_offset = coord_striding + (block_size * coord_stride * block_id)
    x = tl.load(coord_ptr + coord_row_offset, mask=coord_row_offset < coord_numel)
    y = tl.load(
        coord_ptr + coord_row_offset + 1, mask=coord_row_offset + 1 < coord_numel
    )
    z = tl.load(
        coord_ptr + coord_row_offset + 2, mask=coord_row_offset + 2 < coord_numel
    )
    CONST000 = 2.64575131106459
    # ... (omitted intermediate calculations for brevity)
    CONST175 = -0.0738332555375779
    OUT_STRIDING = tl.arange(0, block_size) * output_stride
    OUT_ROW_OFFSET = (
        OUT_STRIDING + (block_size * output_stride * block_id) + col_offset
    )
    Y20 = (
        CONST000 * x * z**2
        + CONST001 * x**3
        + CONST002 * x * y**2
        + CONST003 * y * z**2
    )
    # ... (omitted additional outputs for brevity)
    tl.store(output_ptr + OUT_ROW_OFFSET, Y20, mask=OUT_ROW_OFFSET < output_numel)
    # ... (omitted additional stores for brevity)

@triton.jit
def third_order_bwd(
    coord_ptr: tl.tensor,
    coord_grad_ptr: tl.tensor,
    sph_grad_ptr: tl.tensor,
    block_size: tl.constexpr,
    coord_numel: tl.constexpr,
    output_numel: tl.constexpr,
    col_offset: tl.constexpr,
    output_stride: tl.constexpr,
):
    block_id = tl.program_id(0)
    coord_stride = 3
    coord_striding = tl.arange(0, block_size) * coord_stride
    coord_row_offset = coord_striding + (block_size * coord_stride * block_id)
    x = tl.load(coord_ptr + coord_row_offset, mask=coord_row_offset < coord_numel)
    y = tl.load(
        coord_ptr + coord_row_offset + 1, mask=coord_row_offset + 1 < coord_numel
    )
    z = tl.load(
        coord_ptr + coord_row_offset + 2, mask=coord_row_offset + 2 < coord_numel
    )
    OUT_STRIDING = tl.arange(0, block_size) * output_stride
    OUT_ROW_OFFSET = (
        OUT_STRIDING + (block_size * output_stride * block_id) + col_offset
    )
    g_0 = tl.load(
        sph_grad_ptr + OUT_ROW_OFFSET, mask=OUT_ROW_OFFSET < output_numel
    ).to(x.dtype)
    # ... (omitted intermediate calculations for brevity)
    g_6 = tl.load(
        sph_grad_ptr + OUT_ROW_OFFSET + 6, mask=OUT_ROW_OFFSET + 6 < output_numel
    ).to(x.dtype)
    # Calculate gradients w.r.t. the input coordinates
    g_x = (
        CONST000 * z**2 * g_2
        + CONST001 * x**2 * g_2
        + CONST002 * y**2 * g_4
        + CONST003 * x * g_0
        + CONST004 * y * g_6
    )
    g_y = (
        CONST005 * x * y * g_2
        + CONST006 * y * z**2 * g_4
        + CONST007 * x * y * g_6
        + CONST008 * g_1
    )
    g_z = (
        CONST009 * x * z**2 * g_2
        + CONST010 * y * z**2 * g_4
        + CONST011 * z * g_6
    )
    # Store the gradients in the appropriate locations
    coord_grad_striding = tl.arange(0, block_size) * coord_stride
    coord_grad_row_offset = (
        coord_grad_striding + (block_size * coord_stride * block_id)
    )
    tl.store(
        coord_grad_ptr + coord_grad_row_offset,
        g_x,
        mask=coord_grad_row_offset < coord_numel,
    )
    tl.store(
        coord_grad_ptr + coord_grad_row_offset + 1,
        g_y,
        mask=coord_grad_row_offset + 1 < coord_numel,
    )
    tl.store(
        coord_grad_ptr + coord_grad_row_offset + 2,
        g_z,
        mask=coord_grad_row_offset + 2 < coord_numel,
    )
