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
                (*coords.shape[:-1], 12), dtype=coords.dtype, device=coords.device
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
    
    # Constants for fifth-order spherical harmonics
    CONST000 = 1.73205080756888
    CONST001 = 3.16227766016838
    CONST002 = 5.12347538297980
    CONST003 = 6.48074069840786
    CONST004 = 7.81823298787122
    CONST005 = 10.2469507659596
    CONST006 = -2.09165006633519
    CONST007 = -1
    CONST008 = -6.27495019900557
    CONST009 = -3.96862696659689
    CONST010 = -1.62018517460197
    CONST011 = -12.5499003980111
    CONST012 = -10.2469507659596
    CONST013 = -7.93725393319377
    CONST014 = -5.12347538297980
    CONST015 = -3.24037034920393
    CONST016 = -1.62018517460197
    
    # Intermediate variables
    VAR07 = x * x * x
    VAR08 = x * x
    VAR16 = y * y * y
    VAR17 = y * y
    VAR25 = z * z * z
    VAR26 = z * z
    VAR34 = x * x * x * x
    VAR35 = x * x * x * x * x
    VAR42 = y * y * y * y
    VAR43 = y * y * y * y * y
    VAR50 = z * z * z * z
    VAR51 = z * z * z * z * z
    
    # Compute spherical harmonics
    Y00 = CONST016 * VAR35 - CONST014 * VAR51 * x
    Y01 = CONST013 * x * y * z * (VAR08 + VAR17 + VAR26)
    Y02 = CONST012 * VAR35 + x * (CONST005 * VAR42 + CONST012 * VAR51)
    Y03 = CONST011 * VAR43 + y * (CONST005 * VAR34 + CONST011 * VAR50)
    Y04 = CONST010 * VAR51 + z * (CONST005 * VAR42 + CONST010 * VAR34)
    Y05 = CONST009 * y * (CONST007 * VAR34 + VAR50)
    Y06 = -CONST016 * VAR51 + CONST014 * VAR34 * z
    Y07 = CONST008 * x * (VAR42 - VAR50)
    Y08 = CONST007 * y * (VAR34 - VAR50)
    Y09 = CONST006 * z * (VAR34 - VAR42)
    Y10 = CONST004 * x * y * (VAR17 - VAR26)
    Y11 = CONST003 * y * z * (VAR08 - VAR26)
    
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
    tl.store(
        output_ptr + output_row_offset + 7,
        Y07,
        mask=output_row_offset + 7 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 8,
        Y08,
        mask=output_row_offset + 8 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 9,
        Y09,
        mask=output_row_offset + 9 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 10,
        Y10,
        mask=output_row_offset + 10 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 11,
        Y11,
        mask=output_row_offset + 11 < output_numel,
    )

@triton.jit
def fifth_order_bwd(
    coord_ptr: tl.tensor,
    coord_grad_ptr: tl.tensor,
    sph_grad_ptr: tl.tensor
