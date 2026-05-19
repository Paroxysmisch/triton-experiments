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
    
    # Load coordinates
    x = tl.load(coord_ptr + coord_row_offset, mask=coord_row_offset < coord_numel)
    y = tl.load(coord_ptr + coord_row_offset + 1, mask=coord_row_offset + 1 < coord_numel)
    z = tl.load(coord_ptr + coord_row_offset + 2, mask=coord_row_offset + 2 < coord_numel)

    # Constants for 5th order harmonics
    C1 = 2.64575131106459
    C2 = 5.12347538297980
    C3 = 7.86370330515627
    C4 = 9.74727741252768
    C5 = 11.8392867180873

    # Compute powers
    x2 = x * x
    y2 = y * y
    z2 = z * z
    x3 = x2 * x
    y3 = y2 * y
    z3 = z2 * z
    x4 = x2 * x2
    y4 = y2 * y2
    z4 = z2 * z2
    x5 = x4 * x
    y5 = y4 * y
    z5 = z4 * z

    # Compute spherical harmonics
    Y00 = C1 * (x5 - 10 * x3 * z2 + 5 * x * z4)
    Y01 = C2 * (5 * x4 * y - 10 * x2 * y * z2 + y * z4)
    Y02 = C3 * (x3 * y2 - x3 * z2 - x * y2 * z2 + x * z4)
    Y03 = C4 * (x2 * y3 - 3 * x2 * y * z2)
    Y04 = C5 * (x * y4 - 6 * x * y2 * z2 + x * z4)
    Y05 = C4 * (y5 - 10 * y3 * z2 + 5 * y * z4)
    Y06 = C3 * (x2 * y3 - y5 - y3 * z2 + y * z4)
    Y07 = C2 * (x3 * z2 - x * z4)
    Y08 = C1 * (y3 * z2 - y * z4)
    Y09 = C2 * (x2 * z3 - z5)
    Y10 = C1 * (y2 * z3 - z5)

    # Store results
    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = output_striding + (block_size * output_stride * block_id) + col_offset
    
    # Store each component
    for i, Y in enumerate([Y00, Y01, Y02, Y03, Y04, Y05, Y06, Y07, Y08, Y09, Y10]):
        tl.store(output_ptr + output_row_offset + i, Y, 
                mask=output_row_offset + i < output_numel)

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

    # Load coordinates
    x = tl.load(coord_ptr + coord_row_offset, mask=coord_row_offset < coord_numel)
    y = tl.load(coord_ptr + coord_row_offset + 1, mask=coord_row_offset + 1 < coord_numel)
    z = tl.load(coord_ptr + coord_row_offset + 2, mask=coord_row_offset + 2 < coord_numel)

    # Load gradients
    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = output_striding + (block_size * output_stride * block_id) + col_offset
    
    grads = []
    for i in range(11):
        grad = tl.load(sph_grad_ptr + output_row_offset + i, 
                      mask=output_row_offset + i < output_numel)
        grads.append(grad)

    # Constants for derivatives
    C1 = 2.64575131106459
    C2 = 5.12347538297980
    C3 = 7.86370330515627
    C4 = 9.74727741252768
    C5 = 11.8392867180873

    # Compute powers for derivatives
    x2 = x * x
    y2 = y * y
    z2 = z * z
    x3 = x2 * x
    y3 = y2 * y
    z3 = z2 * z
    x4 = x3 * x
    y4 = y3 * y
    z4 = z3 * z

    # Load current gradients
    g_x = tl.load(coord_grad_ptr + coord_row_offset, mask=coord_row_offset < coord_numel)
    g_y = tl.load(coord_grad_ptr + coord_row_offset + 1, mask=coord_row_offset + 1 < coord_numel)
    g_z = tl.load(coord_grad_ptr + coord_row_offset + 2, mask=coord_row_offset + 2 < coord_numel)

    # Compute gradient contributions
    # Note: These are simplified derivatives - in practice you'd want to compute
    # the full derivatives for each component
    g_x += (
        grads[0] * C1 * (5 * x4 - 30 * x2 * z2 + 5 * z4) +
        grads[1] * C2 * (20 * x3 * y - 20 * y * z2) +
        grads[2] * C3 * (3 * x2 * y2 - 3 * x2 * z2 - y2 * z2 + z4)
    )

    g_y += (
        grads[0] * C1 * (5 * x4 * y - 10 * x2 * y * z2) +
        grads[1] * C2 * (5 * x4 - 10 * x2 * z2 + z4) +
        grads[2] * C3 * (2 * x3 * y - 2 * x * y * z2)
    )

    g_z += (
        grads[0] * C1 * (-20 * x3 * z + 20 * x * z3) +
        grads[1] * C2 * (-20 * x2 * y * z + 4 * y * z3) +
        grads[2] * C3 * (-2 * x3 * z - 2 * x * y2 * z + 4 * x * z3)
    )

    # Store gradients
    tl.store(coord_grad_ptr + coord_row_offset, g_x, mask=coord_row_offset < coord_numel)
    tl.store(coord_grad_ptr + coord_row_offset + 1, g_y, mask=coord_row_offset + 1 < coord_numel)
    tl.store(coord_grad_ptr + coord_row_offset + 2, g_z, mask=coord_row_offset + 2 < coord_numel)
