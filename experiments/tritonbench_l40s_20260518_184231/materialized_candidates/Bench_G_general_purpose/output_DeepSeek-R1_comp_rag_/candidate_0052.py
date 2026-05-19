import triton
import triton.language as tl
import paddle
from ..utils import calculate_settings

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def _swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr,
    stride, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_chunk = tl.program_id(1)
    
    col_start = col_chunk * BLOCK_SIZE
    offsets = row_idx * stride + col_start + tl.arange(0, BLOCK_SIZE)
    mask = (col_start + tl.arange(0, BLOCK_SIZE)) < n_cols

    a = tl.load(a_ptr + offsets, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0)
    silu_a = silu(a)
    c = silu_a * b
    tl.store(c_ptr + offsets, c, mask=mask)

@triton.jit
def _swiglu_backward_kernel(
    dc_ptr, a_ptr, b_ptr,
    da_ptr, db_ptr,
    stride, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_chunk = tl.program_id(1)
    
    col_start = col_chunk * BLOCK_SIZE
    offsets = row_idx * stride + col_start + tl.arange(0, BLOCK_SIZE)
    mask = (col_start + tl.arange(0, BLOCK_SIZE)) < n_cols

    dc = tl.load(dc_ptr + offsets, mask=mask, other=0)
    a = tl.load(a_ptr + offsets, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0)

    sig_a = tl.sigmoid(a)
    silu_a = a * sig_a
    dsilu_a = silu_a * (1 - sig_a) + sig_a
    
    da = dc * dsilu_a * b
    db = dc * silu_a
    
    tl.store(da_ptr + offsets, da, mask=mask)
    tl.store(db_ptr + offsets, db, mask=mask)

def swiglu_forward(a, b):
    orig_shape = a.shape
    n_cols = orig_shape[-1]
    a = a.reshape((-1, n_cols))
    b = b.reshape((-1, n_cols))
    c = paddle.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    n_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (n_rows, n_chunks)

    _swiglu_forward_kernel[grid](
        a, b, c,
        a.stride()[-2],
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return c.reshape(orig_shape)

def swiglu_backward(dc, a, b):
    orig_shape = dc.shape
    n_cols = orig_shape[-1]
    dc = dc.reshape((-1, n_cols))
    a = a.reshape((-1, n_cols))
    b = b.reshape((-1, n_cols))
    
    da = paddle.empty_like(a)
    db = paddle.empty_like(b)
    n_rows = dc.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    n_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (n_rows, n_chunks)

    _swiglu_backward_kernel[grid](
        dc, a, b, da, db,
        dc.stride()[-2],
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return da.reshape(orig_shape), db.reshape(orig_shape)
