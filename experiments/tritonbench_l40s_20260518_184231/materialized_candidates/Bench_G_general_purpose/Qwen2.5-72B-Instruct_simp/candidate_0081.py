import torch
import triton
import triton.language as tl

# Triton Kernel: Forward Pass of LayerNorm
@triton.jit
def _layer_norm_fwd_fused(X, W, B, Y, Mean, Rstd, stride, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    X_row = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean
    mean = tl.sum(X_row, axis=0) / N
    
    # Compute variance
    var = tl.sum((X_row - mean) * (X_row - mean), axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize and apply scale and offset
    Y_row = (X_row - mean) * rstd * tl.load(W + cols, mask=mask, other=0.0) + tl.load(B + cols, mask=mask, other=0.0)
    
    # Store results
    tl.store(Y + row * stride + cols, Y_row, mask=mask)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

# Triton Kernel: Backward Pass for Input Gradient
@triton.jit
def _layer_norm_bwd_dx_fused(DY, X, W, B, Mean, Rstd, DX, DW, DB, stride, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    X_row = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    DY_row = tl.load(DY + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    W_row = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    
    # Compute intermediate values
    X_hat = (X_row - mean) * rstd
    dX_hat = DY_row * W_row
    sum_dX_hat = tl.sum(dX_hat, axis=0)
    sum_dX_hat_X_hat = tl.sum(dX_hat * X_hat, axis=0)
    
    # Compute gradients
    DX_row = (dX_hat * rstd) - (sum_dX_hat * rstd / N) - (sum_dX_hat_X_hat * X_hat * rstd * rstd / N)
    
    # Store gradients
    tl.atomic_add(DW + cols, tl.sum(DY_row * X_hat, axis=0), mask=mask)
    tl.atomic_add(DB + cols, tl.sum(DY_row, axis=0), mask=mask)
    tl.store(DX + row * stride + cols, DX_row, mask=mask)

# Triton Kernel: Aggregation of Weight and Bias Gradients
@triton.jit
def _layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, stride, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    DW_row = tl.load(DW + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    DB_row = tl.load(DB + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Aggregate gradients
    tl.atomic_add(FINAL_DW + cols, DW_row, mask=mask)
    tl.atomic_add(FINAL_DB + cols, DB_row, mask=mask)

# PyTorch Wrapper Class
class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, B, eps=1e-5):
        N = X.shape[-1]
        stride = X.stride(0)
        Mean = torch.empty((X.shape[0],), device=X.device, dtype=X.dtype)
        Rstd = torch.empty((X.shape[0],), device=X.device, dtype=X.dtype)
        Y = torch.empty_like(X)
        
        grid = (X.shape[0],)
        _layer_norm_fwd_fused[grid](X, W, B, Y, Mean, Rstd, stride, N, eps, BLOCK_SIZE=1024)
        
        ctx.save_for_backward(X, W, B, Mean, Rstd)
        ctx.eps = eps
        return Y

    @staticmethod
    def backward(ctx, DY):
        X, W, B, Mean, Rstd = ctx.saved_tensors
        eps = ctx.eps
        N = X.shape[-1]
        stride = X.stride(0)
        DX = torch.empty_like(X)
        DW = torch.zeros_like(W)
        DB = torch.zeros_like(B)
        
        grid = (X.shape[0],)
        _layer_norm_bwd_dx_fused[grid](DY, X, W, B, Mean, Rstd, DX, DW, DB, stride, N, BLOCK_SIZE=1024)
        
        FINAL_DW = torch.zeros_like(W)
        FINAL_DB = torch.zeros_like(B)
        grid = (1,)
        _layer_norm_bwd_dwdb[grid](DW, DB, FINAL_DW, FINAL_DB, stride, N, BLOCK_SIZE=1024)
        
        return DX, FINAL_DW, FINAL_DB, None

# Example Usage
if __name__ == "__main__":
    # Create input tensor and parameters
    X = torch.randn(32, 64, device='cuda')
    W = torch.randn(64, device='cuda', requires_grad=True)
    B = torch.randn(64, device='cuda', requires_grad=True)
    
    # Forward pass
    Y = LayerNorm.apply(X, W, B)
    
    # Compute loss (for example, MSE loss)
    target = torch.randn_like(Y)
    loss = torch.nn.functional.mse_loss(Y, target)
    
    # Backward pass
    loss.backward()
    
    # Print gradients
    print(W.grad)
    print(B.grad)
