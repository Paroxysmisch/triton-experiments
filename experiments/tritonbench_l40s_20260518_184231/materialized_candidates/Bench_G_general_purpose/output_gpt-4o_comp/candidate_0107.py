import triton
import triton.language as tl

@triton.jit
def _swiglu_bwd_kernel(X, Y, DX, DY, DOUT, OUT, n_elements, BLOCK_N: tl.constexpr, RECOMPUTE_OUTPUT: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate row and column indices
    row_idx = pid
    col_start = tl.arange(0, BLOCK_N)
    
    # Calculate the linear index
    offset = row_idx * n_elements + col_start

    # Load input tensors with boundary checking
    x = tl.load(X + offset, mask=col_start < n_elements, other=0.0)
    y = tl.load(Y + offset, mask=col_start < n_elements, other=0.0)
    dout = tl.load(DOUT + offset, mask=col_start < n_elements, other=0.0)

    # Recompute output if needed
    if RECOMPUTE_OUTPUT:
        out = x * tl.sigmoid(y)
        tl.store(OUT + offset, out, mask=col_start < n_elements)
    else:
        out = tl.load(OUT + offset, mask=col_start < n_elements, other=0.0)

    # Compute gradients
    sigmoid_y = tl.sigmoid(y)
    dx = dout * sigmoid_y
    dy = dout * x * sigmoid_y * (1 - sigmoid_y)

    # Store the gradients
    tl.store(DX + offset, dx, mask=col_start < n_elements)
    tl.store(DY + offset, dy, mask=col_start < n_elements)


import torch

def _swiglu_bwd(xy, dout, recompute_output=False):
    assert xy.is_contiguous() and dout.is_contiguous(), "Inputs must be contiguous"

    # Reshape for batch processing
    batch_size, n_elements = xy.shape
    xy = xy.view(batch_size, -1)
    dout = dout.view(batch_size, -1)

    # Split xy into x and y
    x, y = xy.chunk(2, dim=1)

    # Prepare output tensors
    dx = torch.empty_like(x)
    dy = torch.empty_like(y)
    out = torch.empty_like(x) if recompute_output else None

    # Define grid
    BLOCK_N = 128  # Example block size, tune for performance
    grid = (batch_size,)

    # Launch Triton kernel
    _swiglu_bwd_kernel[grid](
        x, y, dx, dy, dout, out, n_elements,
        BLOCK_N=BLOCK_N,
        RECOMPUTE_OUTPUT=recompute_output
    )

    # Return gradients and optionally the recomputed output
    if recompute_output:
        return dx, dy, out
    else:
        return dx, dy
