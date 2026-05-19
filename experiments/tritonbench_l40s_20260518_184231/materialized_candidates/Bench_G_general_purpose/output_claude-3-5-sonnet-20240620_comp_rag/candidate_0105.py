import triton
import triton.language as tl
import torch

@triton.jit
def _swiglu_fwd_kernel(
    X_ptr, Y_ptr, OUT_ptr,
    stride, n_cols,
    BLOCK_N: tl.constexpr
):
    # Program ID maps to row index
    pid = tl.program_id(0)
    
    # Compute pointers for current row
    x_ptr = X_ptr + pid * stride 
    y_ptr = Y_ptr + pid * stride
    out_ptr = OUT_ptr + pid * stride
    
    # Create offsets for the block
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n_cols
    
    # Load inputs
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    
    # Compute SwiGLU forward: SiLU(x) * y
    silu = x * tl.sigmoid(x)  
    out = silu * y
    
    # Store result
    tl.store(out_ptr + offs, out, mask=mask)

@triton.jit
def _swiglu_bwd_kernel(
    X_ptr, Y_ptr, DX_ptr, DY_ptr, DOUT_ptr, OUT_ptr,
    stride, n_cols, RECOMPUTE_OUTPUT: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Compute pointers for current row
    x_ptr = X_ptr + pid * stride
    y_ptr = Y_ptr + pid * stride
    dx_ptr = DX_ptr + pid * stride
    dy_ptr = DY_ptr + pid * stride
    dout_ptr = DOUT_ptr + pid * stride
    out_ptr = OUT_ptr + pid * stride if RECOMPUTE_OUTPUT else None
    
    # Create offsets and mask
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n_cols
    
    # Load inputs and gradients
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    dout = tl.load(dout_ptr + offs, mask=mask, other=0.0)
    
    # Compute intermediate values
    sig_x = tl.sigmoid(x)
    silu_x = x * sig_x
    
    # Compute gradients
    # dx = dout * y * (silu_x * (1 - sig_x) + sig_x)
    # dy = dout * silu_x
    dx = dout * y * (silu_x * (1 - sig_x) + sig_x)
    dy = dout * silu_x
    
    # Store gradients
    tl.store(dx_ptr + offs, dx, mask=mask)
    tl.store(dy_ptr + offs, dy, mask=mask)
    
    # Optionally recompute and store output
    if RECOMPUTE_OUTPUT:
        out = silu_x * y
        tl.store(out_ptr + offs, out, mask=mask)

def _swiglu_bwd(x, y, dout, recompute_output=False):
    # Ensure inputs are contiguous
    x = x.contiguous()
    y = y.contiguous()
    dout = dout.contiguous()
    
    # Get shapes
    batch_shape = x.shape[:-1]
    n_cols = x.shape[-1]
    x_flat = x.view(-1, n_cols)
    y_flat = y.view(-1, n_cols)
    dout_flat = dout.view(-1, n_cols)
    
    # Initialize gradient tensors
    dx = torch.empty_like(x_flat)
    dy = torch.empty_like(y_flat)
    out = torch.empty_like(x_flat) if recompute_output else None
    
    # Calculate grid and block sizes
    n_rows = x_flat.shape[0]
    BLOCK_N = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    
    # Launch kernel
    _swiglu_bwd_kernel[grid](
        x_flat, y_flat, dx, dy, dout_flat, out,
        x_flat.stride(0), n_cols,
        RECOMPUTE_OUTPUT=recompute_output,
        BLOCK_N=BLOCK_N,
        num_warps=4
    )
    
    # Reshape outputs to match input dimensions
    dx = dx.view(batch_shape + (n_cols,))
    dy = dy.view(batch_shape + (n_cols,))
    if recompute_output:
        out = out.view(batch_shape + (n_cols,))
        return dx, dy, out
    return dx, dy
