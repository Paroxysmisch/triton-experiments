import triton
import triton.language as tl
import torch

@triton.jit
def _swiglu_bwd_kernel(
    # Pointers to matrices
    X_ptr, Y_ptr, DOUT_ptr, DX_ptr, DY_ptr,
    # Matrix dimensions
    M, N,
    # Strides for accessing memory
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_doutm, stride_doutn,
    stride_dxm, stride_dxn,
    stride_dym, stride_dyn,
    # Block size
    BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Number of columns per block
    n_cols = N // BLOCK_SIZE
    
    # Row and column index
    row = pid // n_cols
    col = (pid % n_cols) * BLOCK_SIZE
    
    # Compute memory offsets
    x_offset = row * stride_xm + col * stride_xn
    y_offset = row * stride_ym + col * stride_yn
    dout_offset = row * stride_doutm + col * stride_doutn
    dx_offset = row * stride_dxm + col * stride_dxn
    dy_offset = row * stride_dym + col * stride_dyn
    
    # Load data
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < (N - col)
    
    x = tl.load(X_ptr + x_offset + col_offsets * stride_xn, mask=mask)
    y = tl.load(Y_ptr + y_offset + col_offsets * stride_yn, mask=mask)
    dout = tl.load(DOUT_ptr + dout_offset + col_offsets * stride_doutn, mask=mask)
    
    # Compute sigmoid(x)
    sigmoid_x = 1 / (1 + tl.exp(-x))
    
    # Compute gradients
    # dx = dout * y * (sigmoid(x) * (1 - sigmoid(x)))
    # dy = dout * sigmoid(x)
    dx = dout * y * sigmoid_x * (1 - sigmoid_x)
    dy = dout * sigmoid_x
    
    # Store results
    tl.store(DX_ptr + dx_offset + col_offsets * stride_dxn, dx, mask=mask)
    tl.store(DY_ptr + dy_offset + col_offsets * stride_dyn, dy, mask=mask)

def _swiglu_bwd(x: torch.Tensor, y: torch.Tensor, dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for SwiGLU activation function.
    
    Args:
        x: Input tensor for sigmoid gate
        y: Input tensor for linear unit
        dout: Gradient from upstream
        
    Returns:
        tuple: (dx, dy) gradients for x and y
    """
    assert x.is_contiguous(), "Input x must be contiguous"
    assert y.is_contiguous(), "Input y must be contiguous"
    assert dout.is_contiguous(), "Input dout must be contiguous"
    
    batch_size, hidden_dim = x.shape
    
    # Allocate output tensors
    dx = torch.empty_like(x)
    dy = torch.empty_like(y)
    
    # Define block size (can be tuned for performance)
    BLOCK_SIZE = 128
    
    # Calculate grid dimensions
    grid = (batch_size * (hidden_dim + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel
    _swiglu_bwd_kernel[grid](
        x, y, dout, dx, dy,
        batch_size, hidden_dim,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        dout.stride(0), dout.stride(1),
        dx.stride(0), dx.stride(1),
        dy.stride(0), dy.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return dx, dy
