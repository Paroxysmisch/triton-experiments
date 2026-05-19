import triton
import triton.language as tl
import torch

@triton.jit
def _swiglu_bwd_kernel(
    x_ptr, y_ptr,           # input tensors
    dout_ptr,               # gradient from upstream
    dx_ptr, dy_ptr,         # output gradients
    stride,                 # stride between rows
    n_cols: tl.constexpr,   # number of columns
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID gives the row number
    pid = tl.program_id(0)
    
    # Compute pointers to current row
    x_row_ptr = x_ptr + pid * stride
    y_row_ptr = y_ptr + pid * stride
    dout_row_ptr = dout_ptr + pid * stride
    dx_row_ptr = dx_ptr + pid * stride
    dy_row_ptr = dy_ptr + pid * stride
    
    # Create offsets for the columns in the block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load data for current row
    x = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0)
    y = tl.load(y_row_ptr + col_offsets, mask=mask, other=0.0)
    dout = tl.load(dout_row_ptr + col_offsets, mask=mask, other=0.0)
    
    # Compute sigmoid and its derivative
    sig_x = tl.sigmoid(x)
    dsig_x = sig_x * (1.0 - sig_x)
    
    # Compute SwiGLU gradients
    # dx = dout * (y * (sig_x + x * dsig_x))
    # dy = dout * (x * sig_x)
    dx = dout * y * (sig_x + x * dsig_x)
    dy = dout * (x * sig_x)
    
    # Store results
    tl.store(dx_row_ptr + col_offsets, dx, mask=mask)
    tl.store(dy_row_ptr + col_offsets, dy, mask=mask)

def _swiglu_bwd(x: torch.Tensor, y: torch.Tensor, dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for SwiGLU activation.
    
    Args:
        x: First input tensor
        y: Second input tensor
        dout: Upstream gradient
        
    Returns:
        dx, dy: Gradients with respect to x and y
    """
    # Get input shape and flatten batch dimensions
    original_shape = x.shape
    n_cols = original_shape[-1]
    batch_shape = original_shape[:-1]
    
    # Reshape inputs to 2D
    x_2d = x.reshape(-1, n_cols)
    y_2d = y.reshape(-1, n_cols)
    dout_2d = dout.reshape(-1, n_cols)
    n_rows = x_2d.shape[0]
    
    # Allocate output tensors
    dx = torch.empty_like(x_2d)
    dy = torch.empty_like(y_2d)
    
    # Calculate block size and number of warps
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    
    # Launch kernel
    _swiglu_bwd_kernel[(n_rows,)](
        x_2d, y_2d,
        dout_2d,
        dx, dy,
        x_2d.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    # Reshape outputs back to original shape
    return dx.reshape(original_shape), dy.reshape(original_shape)
