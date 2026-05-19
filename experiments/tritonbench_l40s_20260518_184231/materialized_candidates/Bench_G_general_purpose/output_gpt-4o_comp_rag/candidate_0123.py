import triton
import torch
from triton import language as tl
from equitriton.utils import calculate_lastdim_num_blocks

class FifthOrderSphericalHarmonic(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        coords: torch.Tensor,
        output_tensor: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        block_size: int = 64,
        col_offset: int = 0,
    ):
        if not isinstance(output_tensor, torch.Tensor):
            output_tensor = torch.empty(
                (*coords.shape[:-1], 11), dtype=coords.dtype, device=coords.device
            )
        coord_numel = coords.numel()
        output_numel = output_tensor.numel()
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        fifth_order_fwd[num_blocks,](
            coords,
            output_tensor,
            block_size,
            coord_numel,
            output_numel,
            col_offset,
            output_tensor.stride(-2),
        )
        ctx.save_for_backward(coords)
        return output_tensor

    @staticmethod
    def backward(
        ctx,
        sph_grad_tensor: torch.Tensor,
        coord_grad_output: torch.Tensor | None = None,
        block_size: int = 64,
        col_offset: int = 0,
    ) -> torch.Tensor:
        (coords,) = ctx.saved_tensors
        if not isinstance(coord_grad_output, torch.Tensor):
            coord_grad_output = torch.zeros_like(coords)
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        fifth_order_bwd[num_blocks,](
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
def fifth_order_fwd(
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
    # Define constants for fifth-order spherical harmonics
    CONST_A = 2.64575131106459
    CONST_B = 5.12347538297980
    CONST_C = 6.48074069840786
    CONST_D = 10.2469507659596
    CONST_E = -2.09165006633519
    CONST_F = -1
    CONST_G = -6.27495019900557
    CONST_H = -3.96862696659689
    CONST_I = -1.62018517460197
    # Compute terms for spherical harmonics
    VAR_X3 = x * x * x
    VAR_X2 = x * x
    VAR_Y3 = y * y * y
    VAR_Y2 = y * y
    VAR_Z3 = z * z * z
    VAR_Z2 = z * z
    # Compute fifth-order spherical harmonics
    Y00 = CONST_E * VAR_X3 - CONST_G * VAR_Z2 * x
    Y01 = CONST_D * x * y * z
    Y02 = CONST_I * VAR_X3 + x * (CONST_C * VAR_Y2 + CONST_I * VAR_Z2)
    Y03 = CONST_A * VAR_Y3 + CONST_H * VAR_X2 * y + CONST_H * VAR_Z2 * y
    Y04 = CONST_I * VAR_Z3 + z * (CONST_C * VAR_Y2 + CONST_I * VAR_X2)
    Y05 = CONST_B * y * (CONST_F * VAR_X2 + VAR_Z2)
    Y06 = -CONST_E * VAR_Z3 + CONST_G * VAR_X2 * z
    # Store results in output tensor
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

@triton.jit
def fifth_order_bwd(
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
    # Define constants for gradients
    CONST_A = 6.48074069840786
    CONST_B = 12.9614813968157
    CONST_C = -3.96862696659689
    CONST_D = -12.5499003980111
    CONST_E = -10.2469507659596
    CONST_F = -7.93725393319377
    CONST_G = -6.27495019900557
    CONST_H = -5.12347538297980
    CONST_I = -4.86055552380590
    CONST_J = -3.24037034920393
    CONST_K = -1.62018517460197
    VAR_X2 = x * x
    VAR_Y2 = y * y
    VAR_Z2 = z * z
    g_x = tl.load(
        coord_grad_ptr + coord_row_offset, mask=coord_row_offset < coord_numel
    )
    g_y = tl.load(
        coord_grad_ptr + coord_row_offset + 1, mask=coord_row_offset + 1 < coord_numel
    )
    g_z = tl.load(
        coord_grad_ptr + coord_row_offset + 2, mask=coord_row_offset + 2 < coord_numel
    )
    g
