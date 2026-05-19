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
    y = tl.load(coord_ptr + coord_row_offset + 1, mask=coord_row_offset + 1 < coord_numel)
    z = tl.load(coord_ptr + coord_row_offset + 2, mask=coord_row_offset + 2 < coord_numel)

    # Constants for 5th order spherical harmonics
    C1 = 2.50334294 
    C2 = 1.77013077
    C3 = 0.94617469
    C4 = 0.66904654
    C5 = 0.10578554
    C6 = 0.47308734
    C7 = 1.77013077
    C8 = 0.62583573
    C9 = 0.31539156
    C10 = 0.54627422
    C11 = 0.59004358

    x2 = x * x
    y2 = y * y
    z2 = z * z
    x3 = x2 * x
    y3 = y2 * y
    z3 = z2 * z
    x4 = x2 * x2
    y4 = y2 * y2
    z4 = z2 * z2

    Y00 = C1 * (x4 * x - 10 * x3 * z2 + 5 * x * z4)
    Y01 = C2 * y * z * (5 * x2 - z2)
    Y02 = C3 * (x2 - z2) * (7 * x2 - y2 - z2)
    Y03 = C4 * x * y * z * (x2 - y2)
    Y04 = C5 * (x4 - 6 * x2 * y2 + y4)
    Y05 = C6 * y * z * (5 * z2 - 3 * x2 - y2)
    Y06 = C7 * x * z * (4 * z2 - x2 - y2)
    Y07 = C8 * y * (2 * z2 - 3 * x2 - y2)
    Y08 = C9 * x * y * (x2 - y2)
    Y09 = C10 * z * (4 * z2 - 3 * x2 - 3 * y2)
    Y10 = C11 * (x2 + y2 - 2 * z2)

    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = output_striding + (block_size * output_stride * block_id) + col_offset

    tl.store(output_ptr + output_row_offset, Y00, mask=output_row_offset < output_numel)
    tl.store(output_ptr + output_row_offset + 1, Y01, mask=output_row_offset + 1 < output_numel)
    tl.store(output_ptr + output_row_offset + 2, Y02, mask=output_row_offset + 2 < output_numel)
    tl.store(output_ptr + output_row_offset + 3, Y03, mask=output_row_offset + 3 < output_numel)
    tl.store(output_ptr + output_row_offset + 4, Y04, mask=output_row_offset + 4 < output_numel)
    tl.store(output_ptr + output_row_offset + 5, Y05, mask=output_row_offset + 5 < output_numel)
    tl.store(output_ptr + output_row_offset + 6, Y06, mask=output_row_offset + 6 < output_numel)
    tl.store(output_ptr + output_row_offset + 7, Y07, mask=output_row_offset + 7 < output_numel)
    tl.store(output_ptr + output_row_offset + 8, Y08, mask=output_row_offset + 8 < output_numel)
    tl.store(output_ptr + output_row_offset + 9, Y09, mask=output_row_offset + 9 < output_numel)
    tl.store(output_ptr + output_row_offset + 10, Y10, mask=output_row_offset + 10 < output_numel)

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
    y = tl.load(coord_ptr + coord_row_offset + 1, mask=coord_row_offset + 1 < coord_numel)
    z = tl.load(coord_ptr + coord_row_offset + 2, mask=coord_row_offset + 2 < coord_numel)

    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = output_striding + (block_size * output_stride * block_id) + col_offset

    g = [tl.load(sph_grad_ptr + output_row_offset + i, mask=output_row_offset + i < output_numel) for i in range(11)]

    # Constants for 5th order spherical harmonics gradients
    C1 = 12.51671470
    C2 = 8.85065385
    C3 = 6.62322283
    C4 = 2.67618616
    C5 = 0.42314216
    C6 = 2.36543670
    C7 = 8.85065385
    C8 = 1.87750719
    C9 = 1.26156624
    C10 = 2.18509688
    C11 = 1.18008716

    x2 = x * x
    y2 = y * y
    z2 = z * z
    x3 = x2 * x
    y3 = y2 * y
    z3 = z2 * z

    g_x = tl.load(coord_grad_ptr + coord_row_offset, mask=coord_row_offset < coord_numel)
    g_y = tl.load(coord_grad_ptr + coord_row_offset + 1, mask=coord_row_offset + 1 < coord_numel)
    g_z = tl.load(coord_grad_ptr + coord_row_offset + 2, mask=coord_row_offset + 2 < coord_numel)

    # Gradient calculations for x, y, and z
    g_x += (
        C1 * g[0] * (5 * x3 * x - 30 * x2 * z2 + 5 * z4) +
        C2 * g[1] * y * z * 10 * x +
        C3 * g[2] * (4 * x * (7 * x2 - y2 - z2) + 2 * x * (x2 - z2)) +
        C4 * g[3] * y * z * (3 * x2 - y2) +
        C5 * g[4] * (4 * x3 - 12 * x * y2) +
        C6 * g[5] * y * z * (-6 * x) +
        C7 * g[6] * z * (4 * z2 - 3 * x2 - y2) +
        C8 * g[7] * y * (-6 * x) +
        C9 * g[8] * y * (3 * x2 - y2) +
        C10 * g[9] * z * (-6 * x) +
        C11 * g[10] * (2 * x)
    )

    g_y += (
        C2 * g[1] * z * (5 * x2 - z2) +
        C3 * g[2] * (-2 * y * (7 * x2 - y2 - z2)) +
        C4 * g[3] * x * z * (-2 * y) +
        C5 * g[4] * (-12 * x2 * y + 4 * y3) +
        C6 * g[5] * z * (5 * z2 - 3 * x2 - 3 * y2) +
        C7 * g[6] * x * z * (-2 * y) +
        C8 * g[7] * (2 * z2 - 3 * x2 - 3 * y2) +
        C9 * g[8] * x * (x2 - 3 * y2) +
        C10 * g[9] * z * (-6 * y) +
        C11 * g[10] * (2 * y)
    )

    g_z += (
        C1 * g[0] * (-20 * x3 * z + 20 * x * z3) +
        C2 * g[1] * y * (5 * x2 - 3 * z2) +
        C3 * g[2] * (-2 * z * (7 * x2 - y2 - z2) - 2 * z * (x2 - z2)) +
        C4 * g[3] * x * y * (x2 - y2) +
        C6 * g[5] * y * (15 * z2 - 3 * x2 - 3 * y2) +
        C7 * g[6] * x * (12 * z2 - x2 - y2) +
        C8 * g[7] * y * (4 * z) +
        C10 * g[9] * (12 * z2 - 3 * x2 - 3 * y2) +
        C11 * g[10] * (-4 * z)
    )

    tl.store(coord_grad_ptr + coord_row_offset, g_x, mask=coord_row_offset < coord_numel)
    tl.store(coord_grad_ptr + coord_row_offset + 1, g_y, mask=coord_row_offset + 1 < coord_numel)
    tl.store(coord_grad_ptr + coord_row_offset + 2, g_z, mask=coord_row_offset + 2 < coord_numel)
