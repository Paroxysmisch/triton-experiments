import triton
import triton.language as tl
import torch

@triton.jit
def _l2_norm_bwd_kernel(
    X, DY, DX, M, N, stride_x_row, eps, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x_ptr = tl.make_block_ptr(
        base=X + row * stride_x_row,
        shape=(1, N),
        strides=(1, 1),
        offsets=(0, 0),
        block_shape=(1, BLOCK_N),
        order=(0, 1),
    )
    dy_ptr = tl.make_block_ptr(
        base=DY,
        shape=(M, N),
        strides=(stride_x_row, 1),
        offsets=(row * stride_x_row, 0),
        block_shape=(1, BLOCK_N),
        order=(0, 1),
    )

    x = tl.load(x_ptr, boundary_check=(0, 1)).to(tl.float32)
    dy = tl.load(dy_ptr, boundary_check=(0, 1)).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = tl.math.rsqrt(var + eps)

    dx = dy * rstd - tl.sum(dy * x, axis=0) * (1 / (var + eps)) * rstd * x

    if DX is not None:
        dx_ptr = tl.make_block_ptr(
            base=DX + row * stride_x_row,
            shape=(1, N),
            strides=(1, 1),
            offsets=(0, 0),
            block_shape=(1, BLOCK_N),
            order=(0, 1),
        )
        tl.store(dx_ptr, dx.to(x.dtype), boundary_check=(0, 1))


def _l2_norm_bwd(x, dy, eps=1e-5, out=None):
    M, N = dy.shape
    stride_x_row = x.stride(0)
    x = x.reshape(M * N)
    dy = dy.reshape(M * N)
    stride_x_row = x.stride(0)

    if out is None:
        dx = torch.empty_like(dy)
    else:
        dx = out

    assert x.is_contiguous()
    assert dy.is_contiguous()
    assert dx.is_contiguous()

    BLOCK_N = triton.next_power_of_2(x.shape[1])
    if BLOCK_N > 2047:
        raise ValueError("N must be smaller than 2048")

    _l2_norm_bwd_kernel[(M,)](
        x, dy, dx, M, N, stride_x_row, eps, BLOCK_N=BLOCK_N
    )
    return dx.reshape_as(dy)
