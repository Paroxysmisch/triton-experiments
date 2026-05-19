import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X,
    Y,
    stride,
    eps,
    row_block_size,
    feature_size,
    N_CTX: tl.constexpr,
    N_CTX_PADDED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * stride
    row_end = row_start + feature_size
    cols = tl.arange(0, N_CTX)
    mask = cols < feature_size
    x = tl.load(X + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    x_square_sum = tl.sum(x * x)
    var = x_square_sum / feature_size
    rstd = 1 / tl.sqrt(var + eps)
    y = x * rstd
    tl.store(Y + row_start + cols, y, mask=mask)

def _l2_norm_fwd(x, eps=1e-5):
    shape = x.shape
    feature_size = shape[-1]
    assert (
        feature_size <= 65536
    ), "This operation is not supported because the feature dimension size is too large."
    x = x.reshape(-1, feature_size)
    shape = x.shape
    num_rows = shape[0]
    x_ = x.reshape(num_rows, -1, shape[-1])
    row_block_size = x_.shape[1]
    N_CTX = triton.next_power_of_2(row_block_size)
    N_CTX_PADDED = triton.next_power_of_2(row_block_size)
    BLOCK_SIZE = triton.next_power_of_2(feature_size)
    y = torch.empty_like(x)
    grid = (num_rows,)
    _l2_norm_fwd_1pass_kernel[grid](
        x,
        y,
        x.stride(0),
        eps,
        row_block_size,
        feature_size,
        N_CTX=N_CTX,
        N_CTX_PADDED=N_CTX_PADDED,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
    )
    y = y.reshape(shape)
    return y

@triton.jit
def _l2_norm_bwd_kernel(
    X,
    DY,
    DX,
    stride,
    eps,
    row_block_size,
    feature_size,
    N_CTX: tl.constexpr,
    N_CTX_PADDED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * stride
    row_end = row_start + feature_size
    cols = tl.arange(0, N_CTX)
    mask = cols < feature_size
    x = tl.load(X + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    x_square_sum = tl.sum(x * x)
    var = x_square_sum / feature_size
    rstd = 1 / tl.sqrt(var + eps)
    dy = tl.load(DY + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    dx = dy * rstd
    tl.store(DX + row_start + cols, dx, mask=mask)

def _l2_norm_bwd(x, dy, eps=1e-5):
    shape = x.shape
    feature_size = shape[-1]
    assert (
        feature_size <= 65536
    ), "This operation is not supported because the feature dimension size is too large."
    x = x.reshape(-1, feature_size)
    shape = x.shape
    num_rows = shape[0]
    x_ = x.reshape(num_rows, -1, shape[-1])
    row_block_size = x_.shape[1]
    N_CTX = triton.next_power_of_2(row_block_size)
    N_CTX_PADDED = triton.next_power_of_2(row_block_size)
    BLOCK_SIZE = triton.next_power_of_2(feature_size)
    dx = torch.empty_like(x)
    grid = (num_rows,)
    _l2_norm_bwd_kernel[grid](
        x,
        dy,
        dx,
        x.stride(0),
        eps,
        row_block_size,
        feature_size,
        N_CTX=N_CTX,
        N_CTX_PADDED=N_CTX_PADDED,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
    )
    dx = dx.reshape(shape)
    return dx

class L2Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps=1e-5):
        y = _l2_norm_fwd(x, eps)
        ctx.eps = eps
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        x = ctx.saved_tensors[0]
        dx = _l2_norm_bwd(x, dy, ctx.eps)
        return dx, None

def l2_norm(x, eps=1e-5):
    return L2Norm.apply(x, eps)
