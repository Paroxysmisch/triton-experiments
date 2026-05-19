import torch
import triton
import triton.language as tl

# Triton kernel for forward pass of layer normalization
@triton.jit
def _layer_norm_forward_kernel(X_ptr, W_ptr, B_ptr, Y_ptr, Mean_ptr, RSTD_ptr, 
                               n_cols, eps, BLOCK_SIZE: tl.constexpr):
    # Compute row index
    row_idx = tl.program_id(0)
    
    # Load input row
    X = tl.load(X_ptr + row_idx * n_cols + tl.arange(0, BLOCK_SIZE))
    
    # Compute mean
    mean = tl.sum(X, axis=0) / n_cols
    
    # Compute variance
    var = tl.sum((X - mean) ** 2, axis=0) / n_cols
    
    # Compute reciprocal of standard deviation
    rstd = 1 / tl.sqrt(var + eps)
    
    # Normalize
    Y = (X - mean) * rstd
    
    # Apply weight and bias
    W = tl.load(W_ptr + tl.arange(0, BLOCK_SIZE))
    B = tl.load(B_ptr + tl.arange(0, BLOCK_SIZE))
    Y = Y * W + B
    
    # Store results
    tl.store(Y_ptr + row_idx * n_cols + tl.arange(0, BLOCK_SIZE), Y)
    tl.store(Mean_ptr + row_idx, mean)
    tl.store(RSTD_ptr + row_idx, rstd)

# Triton kernel for backward pass of layer normalization
@triton.jit
def _layer_norm_backward_kernel(dY_ptr, X_ptr, W_ptr, Mean_ptr, RSTD_ptr, 
                                dX_ptr, dW_ptr, dB_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    # Compute row index
    row_idx = tl.program_id(0)
    
    # Load data
    dY = tl.load(dY_ptr + row_idx * n_cols + tl.arange(0, BLOCK_SIZE))
    X = tl.load(X_ptr + row_idx * n_cols + tl.arange(0, BLOCK_SIZE))
    W = tl.load(W_ptr + tl.arange(0, BLOCK_SIZE))
    mean = tl.load(Mean_ptr + row_idx)
    rstd = tl.load(RSTD_ptr + row_idx)
    
    # Compute dX
    dX = W * rstd * (dY - tl.sum(dY, axis=0) / n_cols - (X - mean) * rstd * tl.sum(dY * (X - mean), axis=0) / n_cols)
    tl.store(dX_ptr + row_idx * n_cols + tl.arange(0, BLOCK_SIZE), dX)
    
    # Compute dW and dB
    dW = tl.sum(dY * (X - mean) * rstd, axis=0)
    dB = tl.sum(dY, axis=0)
    
    # Store gradients
    tl.atomic_add(dW_ptr + tl.arange(0, BLOCK_SIZE), dW)
    tl.atomic_add(dB_ptr + tl.arange(0, BLOCK_SIZE), dB)

def layer_norm_forward(X, W, B, eps):
    n_rows, n_cols = X.shape
    BLOCK_SIZE = 128
    Y = torch.empty_like(X)
    Mean = torch.empty(n_rows, device=X.device)
    RSTD = torch.empty(n_rows, device=X.device)
    
    grid = (n_rows,)
    _layer_norm_forward_kernel[grid](X, W, B, Y, Mean, RSTD, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    
    return Y, Mean, RSTD

def layer_norm_backward(dY, X, W, Mean, RSTD):
    n_rows, n_cols = X.shape
    BLOCK_SIZE = 128
    dX = torch.empty_like(X)
    dW = torch.zeros_like(W)
    dB = torch.zeros_like(B)
    
    grid = (n_rows,)
    _layer_norm_backward_kernel[grid](dY, X, W, Mean, RSTD, dX, dW, dB, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return dX, dW, dB

class LigerLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, B, eps):
        X = X.contiguous()
        W = W.contiguous()
        B = B.contiguous()
        
        Y, Mean, RSTD = layer_norm_forward(X, W, B, eps)
        
        ctx.save_for_backward(X, W, Mean, RSTD)
        ctx.eps = eps
        
        return Y

    @staticmethod
    def backward(ctx, dY):
        X, W, Mean, RSTD = ctx.saved_tensors
        eps = ctx.eps
        
        dX, dW, dB = layer_norm_backward(dY, X, W, Mean, RSTD)
        
        return dX, dW, dB, None

# Example usage:
# X, W, B = torch.randn(64, 128, device='cuda'), torch.randn(128, device='cuda'), torch.randn(128, device='cuda')
# eps = 1e-5
# Y = LigerLayerNormFunction.apply(X, W, B, eps)
