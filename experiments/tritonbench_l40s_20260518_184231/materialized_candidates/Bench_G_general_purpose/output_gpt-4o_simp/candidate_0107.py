import triton
import triton.language as tl

@triton.jit
def _swiglu_bwd_kernel(X, Y, DOUT, DX, DY, OUT, stride_x, stride_y, stride_dout, stride_dx, stride_dy, stride_out, n_cols, BLOCK_SIZE: tl.constexpr):
    # Program ID
    pid = tl.program_id(axis=0)

    # Offsets
    row_start = pid * BLOCK_SIZE
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Load data
    X_ptrs = X + row_start * stride_x + col_offsets
    Y_ptrs = Y + row_start * stride_y + col_offsets
    DOUT_ptrs = DOUT + row_start * stride_dout + col_offsets

    # Conditionally load OUT if it's not None
    if OUT is not None:
        OUT_ptrs = OUT + row_start * stride_out + col_offsets
        out = tl.load(OUT_ptrs, mask=col_offsets < n_cols, other=0.0)
    else:
        out = None

    x = tl.load(X_ptrs, mask=col_offsets < n_cols, other=0.0)
    y = tl.load(Y_ptrs, mask=col_offsets < n_cols, other=0.0)
    dout = tl.load(DOUT_ptrs, mask=col_offsets < n_cols, other=0.0)

    # Compute sigmoid
    sigmoid_y = 1 / (1 + tl.exp(-y))

    # Compute gradients
    dx = dout * y * sigmoid_y * (1 + x * (1 - sigmoid_y))
    dy = dout * x * sigmoid_y * (1 - sigmoid_y)

    # Store results
    DX_ptrs = DX + row_start * stride_dx + col_offsets
    DY_ptrs = DY + row_start * stride_dy + col_offsets

    tl.store(DX_ptrs, dx, mask=col_offsets < n_cols)
    tl.store(DY_ptrs, dy, mask=col_offsets < n_cols)

import torch

def _swiglu_bwd(X, Y, DOUT, OUT=None):
    assert X.shape == Y.shape == DOUT.shape, "Input tensors must have the same shape"

    # Reshape inputs to ensure contiguous memory layout
    X = X.contiguous()
    Y = Y.contiguous()
    DOUT = DOUT.contiguous()
    if OUT is not None:
        OUT = OUT.contiguous()

    # Get dimensions
    n_rows, n_cols = X.shape

    # Allocate output tensors
    DX = torch.empty_like(X)
    DY = torch.empty_like(Y)

    # Define grid and block size
    BLOCK_SIZE = 128  # This can be tuned based on the GPU architecture
    grid = (triton.cdiv(n_rows, BLOCK_SIZE),)

    # Launch the Triton kernel
    _swiglu_bwd_kernel[grid](
        X, Y, DOUT, DX, DY, OUT,
        X.stride(0), Y.stride(0), DOUT.stride(0), DX.stride(0), DY.stride(0), OUT.stride(0) if OUT is not None else 0,
        n_cols, BLOCK_SIZE=BLOCK_SIZE
    )

    return DX, DY
