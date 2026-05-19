import torch
import triton
import triton.language as tl
from .utils import calculate_settings

@triton.autotune(configs=calculate_settings(), key=['dim'])
@triton.jit
def _log_softmax_linear_forward(x, A, b, out, stride, n_elements, D: tl.constexpr, K: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + arange
    mask = offsets < n_elements
    batch_offsets = offsets // D
    dim_offsets = offsets % D
    batch_mask = batch_offsets < batch_size
    ptr = x + batch_offsets * stride + dim_offsets
    x_row = tl.load(ptr, mask=batch_mask & mask, other=-float('inf')).to(tl.float32)
    A_col = tl.load(A + dim_offsets[:, None] * K, mask=mask[:, None], other=float(0)).to(tl.float32)
    if bias is not None:
        b_ptr = b + offsets
        b_val = tl.load(b_ptr, mask=mask, other=float(0)).to(tl.float32)
        A_row_b = tl.sum(x_row * A_col + b_val, axis=1)
    else:
        A_row_b = tl.sum(x_row * A_col, axis=1)
    max_val = tl.max(A_row_b, axis=0)
    exp_sum = tl.sum(tl.exp(A_row_b - max_val))
    Z = tl.log(exp_sum) + max_val
    out_val = A_row_b - Z
    tl.store(out + offsets, out_val, mask=mask)

def log_softmax_linear_forward(x, A, b, dim, dtype):
    global _log_softmax_linear_forward
    assert x.is_contiguous()
    assert A.is_contiguous()
    assert b is not None and b.is_contiguous() or b is None
    assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
    if dtype is not None:
        assert dtype == x.dtype, "Casting input from {} to {} is not supported".format(x.dtype, dtype)
    else:
        dtype = x.dtype
    x_shape = list(x.shape)
    n_dims = len(x_shape)
    dim = dim % n_dims
    x_dim_size = x_shape[dim]
    del x_shape[dim]
    M = int(numpy.prod(x_shape))
    out = torch.empty((M,), device=x.device, dtype=dtype)
    grid = lambda opt: (M, 1, 1)
    BLOCK_SIZE = 256
    _log_softmax_linear_forward[grid](x, A, b, out, x_dim_size, M, D=x_dim_size, K=A.size(1), BLOCK_SIZE=BLOCK_SIZE)
    out = out.reshape(x_shape)
    return out
