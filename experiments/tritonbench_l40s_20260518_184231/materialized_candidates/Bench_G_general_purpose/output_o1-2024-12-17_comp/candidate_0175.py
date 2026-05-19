import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_bwd_kernel(
    X, DY, DX,
    M, N,
    stride_x_row, stride_dy_row, stride_dx_row,
    eps,
    BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    x_row_ptr = X + pid_m * stride_x_row + offsets
    dy_row_ptr = DY + pid_m * stride_dy_row + offsets
    dx_row_ptr = DX + pid_m * stride_dx_row + offsets
    mask = offsets < N

    x = tl.load(x_row_ptr, mask=mask, other=0.0)
    dy = tl.load(dy_row_ptr, mask=mask, other=0.0)

    sum_x2 = tl.sum(x * x, axis=0)
    sum_dyx = tl.sum(dy * x, axis=0)

    rstd = 1.0 / tl.sqrt(sum_x2 + eps)
    factor = sum_dyx * (1.0 / (sum_x2 + eps)) * rstd
    dx = dy * rstd - factor * x
    tl.store(dx_row_ptr, dx, mask=mask)

def _l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    assert x.dim() == 2, "Input x must be a 2D tensor"
    assert dy.dim() == 2, "Grad dy must be a 2D tensor"
    M, N = x.shape
    # Find next power of 2
    def next_power_of_2(val):
        return 1 << (val - 1).bit_length()
    BLOCK_N = next_power_of_2(N)
    # Raise error if N exceeds BLOCK_N
    if N > BLOCK_N:
        raise ValueError("N exceeds maximum permissible block size")

    x_contig = x.contiguous()
    dy_contig = dy.contiguous()
    dx = torch.empty_like(x_contig)

    stride_x_row = x_contig.stride(0)
    stride_dy_row = dy_contig.stride(0)
    stride_dx_row = dx.stride(0)

    grid = (M, )
    _l2_norm_bwd_kernel[grid](
        x_contig, dy_contig, dx,
        M, N,
        stride_x_row, stride_dy_row, stride_dx_row,
        eps,
        BLOCK_N=BLOCK_N
    )
    return dx.reshape(x.shape)
