import triton
import torch
from triton import language as tl
from equitriton.utils import calculate_lastdim_num_blocks

# Forward Kernel
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
    CONST002 = 5.12347538297980
    CONST004 = 6.48074069840786
    CONST005 = 10.2469507659596
    CONST006 = -2.09165006633519
    CONST007 = -1
    CONST008 = -6.27495019900557
    CONST009 = -3.96862696659689
    CONST010 = -1.62018517460197
    VAR07 = x * x * x
    VAR08 = x * x
    VAR16 = y * y * y
    VAR17 = y * y
    VAR25 = z * z * z
    VAR26 = z * z
    Y00 = CONST006 * VAR07 - CONST008 * VAR26 * x
    Y01 = CONST005 * x * y * z
    Y02 = CONST010 * VAR07 + x * (CONST004 * VAR17 + CONST010 * VAR26)
    Y03 = CONST000 * VAR16 + CONST009 * VAR08 * y + CONST009 * VAR26 * y
    Y04 = CONST010 * VAR25 + z * (CONST004 * VAR17 + CONST010 * VAR08)
    Y05 = CONST002 * y * (CONST007 * VAR08 + VAR26)
    Y06 = -CONST006 * VAR25 + CONST008 * VAR08 * z
    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = (
        output_striding + (block_size * output_stride * block_id) + col_offset
    )
    tl.store(output_ptr + output_row_offset, Y00, mask=output_row_offset < output_numel)
    tl.store(
        output_ptr + output_row_offset + 1,
        Y01,
        mask=output_row_offset + 1 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 2,
        Y02,
        mask=output_row_offset + 2 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 3,
        Y03,
        mask=output_row_offset + 3 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 4,
        Y04,
        mask=output_row_offset + 4 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 5,
        Y05,
        mask=output_row_offset + 5 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 6,
        Y06,
        mask=output_row_offset + 6 < output_numel,
    )

# Backward Kernel
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
    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = (
        output_striding + (block_size * output_stride * block_id) + col_offset
    )
    g_0 = tl.load(
        sph_grad_ptr + output_row_offset, mask=output_row_offset < output_numel
    )
    g_1 = tl.load(
        sph_grad_ptr + output_row_offset + 1, mask=output_row_offset + 1 < output_numel
    )
    g_2 = tl.load(
        sph_grad_ptr + output_row_offset + 2, mask=output_row_offset + 2 < output_numel
    )
    g_3 = tl.load(
        sph_grad_ptr + output_row_offset + 3, mask=output_row_offset + 3 < output_numel
    )
    g_4 = tl.load(
        sph_grad_ptr + output_row_offset + 4, mask=output_row_offset + 4 < output_numel
    )
    g_5 = tl.load(
        sph_grad_ptr + output_row_offset + 5, mask=output_row_offset + 5 < output_numel
    )
    g_6 = tl.load(
        sph_grad_ptr + output_row_offset + 6, mask=output_row_offset + 6 < output_numel
    )
    CONST002 = 6.48074069840786
    CONST005 = 12.9614813968157
    CONST007 = -3.96862696659689
    CONST008 = -12.5499003980111
    CONST009 = -10.2469507659596
    CONST010 = -7.93725393319377
    CONST011 = -6.27495019900557
    CONST012 = -5.12347538297980
    CONST013 = -4.86055552380590
    CONST014 = -3.24037034920393
    CONST015 = -1.62018517460197
    VAR08 = x * x
    VAR17 = y * y
    VAR26 = z * z
    g_x = tl.load(
        coord_grad_ptr + coord_row_offset, mask=coord_row_offset < coord_numel
    )
    g_y = tl.load(
        coord_grad_ptr + coord_row_offset + 1, mask=coord_row_offset + 1 < coord_numel
    )
    g_z = tl.load(
        coord_grad_ptr + coord_row_offset + 2, mask=coord_row_offset + 2 < coord_numel
    )
    g_x += (
        CONST008 * g_6 * x * z
        - CONST009 * g_1 * y * z
        + CONST009 * g_5 * x * y
        + CONST010 * g_3 * x * y
        + CONST014 * g_4 * x * z
        + g_0 * (CONST011 * VAR08 - CONST011 * VAR26)
        + g_2 * (CONST002 * VAR17 + CONST01
