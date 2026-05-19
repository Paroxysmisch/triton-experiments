import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_kernel(
    X,          # pointer to input tensor [batch, seq_len, hidden_dim]
    W,          # pointer to weight tensor [seq_len, hidden_dim]
    Y,          # pointer to output tensor [batch, seq_len, hidden_dim]
    stride_x_N, # stride for batch dimension of X
    stride_x_hn,# stride for sequence dimension of X
    stride_x_hd,# stride for hidden dimension of X
    stride_y_N, # stride for batch dimension of Y
    stride_y_hn,# stride for sequence dimension of Y
    stride_y_hd,# stride for hidden dimension of Y
    stride_w_hn,# stride for sequence dimension of W
    stride_w_hd,# stride for hidden dimension of W
    N,          # size of hidden dimension
    eps,        # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # size of parallel processing blocks
):
    # Get program ID for batch and sequence dimensions
    batch_idx = tl.program_id(0)  
    seq_idx = tl.program_id(1)

    # Calculate base pointers for current batch/sequence position
    X_ptr = X + batch_idx * stride_x_N + seq_idx * stride_x_hn
    Y_ptr = Y + batch_idx * stride_y_N + seq_idx * stride_y_hn
    W_ptr = W + seq_idx * stride_w_hn

    # Step 1: Calculate mean
    mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean += x
    mean = tl.sum(mean, axis=0) / N

    # Step 2: Calculate variance
    var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        var += x * x
    var = tl.sum(var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Step 3: Normalize and apply weights
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        # Load weights and input values
        w = tl.load(W_ptr + cols, mask=mask).to(tl.float32)
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        
        # Normalize and scale
        x_normalized = (x - mean) * rstd
        y = x_normalized * w
        
        # Store result
        tl.store(Y_ptr + cols, y.to(X.dtype.element_ty), mask=mask)

def layernorm_forward(X, W, eps=1e-5):
    """
    Forward pass for LayerNorm
    Args:
        X: Input tensor of shape [batch, seq_len, hidden_dim]
        W: Weight tensor of shape [seq_len, hidden_dim]
        eps: Small constant for numerical stability
    Returns:
        Y: Normalized tensor of same shape as X
    """
    # Input validation
    assert len(X.shape) == 3, "Input tensor must be 3D"
    assert len(W.shape) == 2, "Weight tensor must be 2D"
    assert X.shape[-1] == W.shape[-1], "Hidden dimensions must match"
    assert X.shape[-2] == W.shape[-2], "Sequence lengths must match"

    # Prepare output tensor
    Y = torch.empty_like(X)

    # Calculate strides for tensor access
    stride_x_N = X.stride(0)
    stride_x_hn = X.stride(1)
    stride_x_hd = X.stride(2)
    
    stride_y_N = Y.stride(0)
    stride_y_hn = Y.stride(1)
    stride_y_hd = Y.stride(2)
    
    stride_w_hn = W.stride(0)
    stride_w_hd = W.stride(1)

    # Set block size for parallel processing
    N = X.shape[-1]
    BLOCK_SIZE = min(128, triton.next_power_of_2(N))

    # Configure grid for kernel launch
    grid = (X.shape[0], X.shape[1])  # (batch_size, seq_len)
    
    # Launch kernel
    _layer_norm_fwd_kernel[grid](
        X, W, Y,
        stride_x_N, stride_x_hn, stride_x_hd,
        stride_y_N, stride_y_hn, stride_y_hd,
        stride_w_hn, stride_w_hd,
        N, eps, BLOCK_SIZE,
    )

    return Y
