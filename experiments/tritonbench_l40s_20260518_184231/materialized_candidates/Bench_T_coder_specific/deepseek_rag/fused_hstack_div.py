import triton
import triton.language as tl
import torch

@triton.jit
def _fused_hstack_div(X_ptr, D_ptr, Y_ptr, n_rows, n_cols, n_divs, rounding_mode):
    pid = tl.program_id(0)
    did = tl.program_id(1)
    row = pid
    col = did
    div = tl.load(D_ptr + col)
    Xs = [tl.load(X_ptr + row + col * n_cols + i * n_rows * n_cols) for i in range(n_divs)]
    X = tl.hstack(Xs)
    Y = X / div
    if rounding_mode == 'trunc':
        Y = tl.floor(Y)
    elif rounding_mode == 'floor':
        Y = tl.floor(Y)
    tl.store(Y_ptr + row + col * n_cols, Y)

def fused_hstack_div(tensors, divisor, rounding_mode=None, out=None):
    if out is None:
        out = torch.empty_like(tensors[0])
    assert out.shape == tensors[0].shape
    assert divisor.shape == () or divisor.shape == tensors[0].shape
    n_rows, n_cols = out.shape
    n_divs = len(tensors)
    grid = (n_rows, n_cols)
    _fused_hstack_div[(grid,)](*[t.data.ptr for t in tensors] + divisor.data.ptr, out.data.ptr, n_rows, n_cols, n_divs, rounding_mode)
    return out
