import triton
import triton.language as tl

@triton.jit
def _swiglu_bwd_kernel(
    X, Y, DOUT, DX, DY, 
    stride_x, stride_y, stride_dout, 
    stride_dx, stride_dy, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data
    x = tl.load(X + offsets, mask=offsets < stride_x, other=0.0)
    y = tl.load(Y + offsets, mask=offsets < stride_y, other=0.0)
    dout = tl.load(DOUT + offsets, mask=offsets < stride_dout, other=0.0)
    
    # Compute sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    
    # Compute the gradient for Y
    dy = dout * sigmoid_x
    # Compute the gradient for X
    dx = dout * y * (sigmoid_x * (1.0 - sigmoid_x))
    
    # Store the results
    tl.store(DX + offsets, dx, mask=offsets < stride_dx)
    tl.store(DY + offsets, dy, mask=offsets < stride_dy)

import torch
import triton
import triton.language as tl

def _swiglu_bwd(X, Y, DOUT, OUT=None):
    # Ensure inputs are contiguous
    X = X.contiguous()
    Y = Y.contiguous()
    DOUT = DOUT.contiguous()
    
    # Get the shapes
    N = X.shape[0]
    BLOCK_SIZE = 1024  # Adjust block size as needed
    
    # Allocate memory for gradients
    DX = torch.empty_like(X)
    DY = torch.empty_like(Y)
    
    # Define the grid
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Launch the kernel
    _swiglu_bwd_kernel[grid](
        X, Y, DOUT, DX, DY,
        X.stride(0), Y.stride(0), DOUT.stride(0),
        DX.stride(0), DY.stride(0),
        BLOCK_SIZE
    )
    
    return DX, DY
