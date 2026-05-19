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
    
    # Fifth-order spherical harmonics constants
    C0 = 0.683184105191914
    C1 = 2.39676839248666
    C2 = 4.79353678497332
    C3 = 1.98431348329845
    C4 = 3.31110374356685
    C5 = 5.83141347
    C6 = 11.4564392
    C7 = 1.11803398874989
    C8 = 4.47213595499958
    C9 = 8.94427190999916
    C10 = 0.559016994374947
    C11 = 2.23606797749979
    
    # Compute fifth-order terms
    x2 = x * x
    y2 = y * y
    z2 = z * z
    x3 = x2 * x
    y3 = y2 * y
    z3 = z2 * z
    x4 = x3 * x
    y4 = y3 * y
    z4 = z3 * z
    x5 = x4 * x
    y5 = y4 * y
    z5 = z4 * z
    
    # Spherical harmonics components
    Y00 = C0 * x * (x4 - 10*x2*y2 + 5*y4)
    Y01 = C1 * y * (5*x4 - 10*x2*y2 + y4)
    Y02 = C2 * z * x * (x2 - 3*y2)
    Y03 = C3 * z * y * (3*x2 - y2)
    Y04 = C4 * x * (x2 + y2 - 8*z2) * z
    Y05 = C5 * y * (x2 + y2 - 8*z2) * z
    Y06 = C6 * z2 * x * y
    Y07 = C7 * z3 * (x2 - y2)
    Y08 = C8 * z4 * x
    Y09 = C9 * z4 * y
    Y10 = C10 * z5 - C11 * z3 * (x2 + y2) + C10 * z * (x2 + y2)**2
    
    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = (
        output_striding + (block_size * output_stride * block_id) + col_offset
    )
    
    for i in range(11):
        tl.store(
            output_ptr + output_row_offset + i,
            locals()[f"Y{i:02d}"],
            mask=output_row_offset + i < output_numel
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
    
    # Load all gradients
    gradients = tl.zeros((11,), tl.float32)
    for i in range(11):
        gradients = tl.where(
            output_row_offset + i < output_numel,
            gradients + tl.load(sph_grad_ptr + output_row_offset + i),
            gradients
        )
    g = [tl.sum(gradients[i]) for i in range(11)]
    
    # Compute derivatives
    x2 = x * x
    y2 = y * y
    z2 = z * z
    x3 = x2 * x
    y3 = y2 * y
    z3 = z2 * z
    x4 = x3 * x
    y4 = y3 * y
    z4 = z3 * z
    
    # Partial derivatives for each harmonic component
    dY00_dx = 0.683184105191914 * (5*x4 - 30*x2*y2 + 5*y4)
    dY00_dy = 0.683184105191914 * x * (-20*x2*y + 20*y3)
    
    dY01_dx = 2.39676839248666 * y * (20*x3 - 20*x*y2)
    dY01_dy = 2.39676839248666 * (5*x4 - 30*x2*y2 + 5*y4)
    
    dY02_dx = 4.79353678497332 * z * (3*x2 - 3*y2)
    dY02_dy = 4.79353678497332 * z * (-6*x*y)
    dY02_dz = 4.79353678497332 * x * (x2 - 3*y2)
    
    dY03_dx = 1.98431348329845 * z * (6*x*y)
    dY03_dy = 1.98431348329845 * z * (3*x2 - 3*y2)
    dY03_dz = 1.98431348329845 * y * (3*x2 - y2)
    
    dY04_dx = 3.31110374356685 * z * (3*x2 + y2 - 8*z2)
    dY04_dy = 3.31110374356685 * z * 2*x*y
    dY04_dz = 3.31110374356685 * x * (x2 + y2 - 24*z2)
    
    dY05_dx = 5.83141347 * z * 2*x*y
    dY05_dy = 5.83141347 * z * (x2 + 3*y2 - 8*z2)
    dY05_dz = 5.83141347 * y * (x2 + y2 - 24*z2)
    
    dY06_dx = 11.4564392 * z2 * y
    dY06_dy = 11.4564392 * z2 * x
    dY06_dz = 11.4564392 * 2*z * x*y
    
    dY07_dx = 1.11803398874989 * 2*z3 * x
    dY07_dy = -1.11803398874989 * 2*z3 * y
    dY07_dz = 1.11803398874989 * 3*z2 * (x2 - y2)
    
    dY08_dx = 4.47213595499958 * z4
    dY08_dz = 4.47213595499958 * 4*z3 * x
    
    dY09_dy = 8.94427190999916 * z4
    dY09_dz = 8.94427190999916 * 4*z3 * y
    
    dY10_dx = 0.559016994374947 * 2*z*(x2 + y2) - 2.23606797749979 * 2*z3*x
    dY10_dy = 0.559016994374947 * 2*z*(x2 + y2) - 2.23606797749979 * 2*z3*y
    dY10_dz = 0.559016994374947 * (5*z4 - 3*z2*(x2 + y2) + (x2 + y2)**2) - 2.23606797749979 * 2*z*(x2 + y2)
    
    # Accumulate gradients
    g_x = (
        g[0] * dY00_dx + g[1] * dY01_dx + g[2] * dY02_dx +
        g[3] * dY03_dx + g[4] * dY04_dx + g[5] * dY05_dx +
        g[6] * dY06_dx + g[7] * dY07_dx + g[8] * dY08_dx +
        g[10] * dY10_dx
    )
    
    g_y = (
        g[0] * dY00_dy + g[1] * dY01_dy + g[2] * dY02_dy +
        g[3] * dY03_dy + g[4] * dY04_dy + g[5] * dY05_dy +
        g[6] * dY06_dy + g[7] * dY07_dy + g[9] * dY09_dy +
        g[10] * dY10_dy
    )
    
    g_z = (
        g[2] * dY02_dz + g[3] * dY03_dz + g[4] * dY04_dz +
        g[5] * dY05_dz + g[6] * dY06_dz + g[7] * dY07_dz +
        g[8] * dY08_dz + g[9] * dY09_dz + g[10] * dY10_dz
    )
    
    # Load existing gradients and accumulate
    existing_gx = tl.load(coord_grad_ptr + coord_row_offset, mask=coord_row_offset < coord_numel)
    existing_gy = tl.load(coord_grad_ptr + coord_row_offset + 1, mask=coord_row_offset + 1 < coord_numel)
    existing_gz = tl.load(coord_grad_ptr + coord_row_offset + 2, mask=coord_row_offset + 2 < coord_numel)
    
    tl.store(
        coord_grad_ptr + coord_row_offset,
        existing_gx + g_x,
        mask=coord_row_offset < coord_numel
    )
    tl.store(
        coord_grad_ptr + coord_row_offset + 1,
        existing_gy + g_y,
        mask=coord_row_offset + 1 < coord_numel
    )
    tl.store(
        coord_grad_ptr + coord_row_offset + 2,
        existing_gz + g_z,
        mask=coord_row_offset + 2 < coord_numel
    )
