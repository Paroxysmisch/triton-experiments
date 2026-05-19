@triton.heuristics({
    "HAS_BIAS": lambda args: args["B"] is not None,
    "HAS_RESIDUAL": lambda args: args["RESIDUAL"] is not None,
    "HAS_X1": lambda args: args["X1"] is not None
})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    # Inputs
    X, Y, W, B,  # Main branch tensors
    RESIDUAL,    # Optional residual connection
    X1, W1, B1,  # Optional second branch tensors
    Mean, Rstd,  # Statistics tensors
    # Strides
    stride_x_row, stride_y_row,
    stride_res_row, stride_x1_row,
    # Dimensions
    M, N, eps,
    # Constants
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_X1: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    # Map program ID to row
    row = tl.program_id(0)
    
    # Compute base pointers
    X_ptr = X + row * stride_x_row
    Y_ptr = Y + row * stride_y_row
    
    # Load input data
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X_ptr + cols, mask=mask, other=0.).to(tl.float32)
    
    # Compute statistics
    if not IS_RMS_NORM:
        # Regular Layer Norm
        mean = tl.sum(x, axis=0) / N
        xbar = tl.where(mask, x - mean, 0.)
        var = tl.sum(xbar * xbar, axis=0) / N
        tl.store(Mean + row, mean)
    else:
        # RMS Norm
        xbar = tl.where(mask, x, 0.)
        var = tl.sum(xbar * xbar, axis=0) / N
        
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    
    # Normalize and transform
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    w = tl.load(W + cols, mask=mask)
    y = x_hat * w
    
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask)
        y += b
        
    if HAS_RESIDUAL:
        res = tl.load(RESIDUAL + row * stride_res_row + cols, mask=mask)
        y += res
        
    # Optional second branch
    if HAS_X1:
        x1 = tl.load(X1 + row * stride_x1_row + cols, mask=mask)
        w1 = tl.load(W1 + cols, mask=mask)
        y1 = x1 * w1
        if B1 is not None:
            b1 = tl.load(B1 + cols, mask=mask)
            y1 += b1
            
    # Store outputs
    tl.store(Y_ptr + cols, y, mask=mask)
