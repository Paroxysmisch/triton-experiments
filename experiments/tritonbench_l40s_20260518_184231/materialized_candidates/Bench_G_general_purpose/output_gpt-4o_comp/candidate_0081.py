import torch
import triton
import triton.language as tl

# Kernel to perform forward pass
@triton.jit
def _layer_norm_fwd_fused(X, W, B, Y, Mean, Rstd, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # Load input data for this block
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offs, mask=offs < X.shape[0])
    
    # Compute mean and variance
    mean = tl.sum(x, axis=0) / BLOCK_SIZE
    var = tl.sum((x - mean) ** 2, axis=0) / BLOCK_SIZE
    rstd = 1 / tl.sqrt(var + 1e-5)
    
    # Normalize and apply learned parameters
    y = (x - mean) * rstd * tl.load(W) + tl.load(B)
    
    # Store results
    tl.store(Y + offs, y, mask=offs < X.shape[0])
    tl.store(Mean + pid, mean)
    tl.store(Rstd + pid, rstd)

# Kernel to compute gradient of inputs
@triton.jit
def _layer_norm_bwd_dx_fused(DY, X, W, Mean, Rstd, DX, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dy = tl.load(DY + offs, mask=offs < DY.shape[0])
    x = tl.load(X + offs, mask=offs < X.shape[0])
    mean = tl.load(Mean + pid)
    rstd = tl.load(Rstd + pid)
    
    # Compute gradients
    dx = (dy - tl.sum(dy, axis=0) / BLOCK_SIZE - (x - mean) * tl.sum(dy * (x - mean), axis=0) * rstd ** 2 / BLOCK_SIZE) * rstd * tl.load(W)
    
    # Store gradients
    tl.store(DX + offs, dx, mask=offs < DX.shape[0])

# Kernel to finalize backward pass
@triton.jit
def _layer_norm_bwd_dwdb(DY, X, Mean, Rstd, DW, DB, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dy = tl.load(DY + offs, mask=offs < DY.shape[0])
    x = tl.load(X + offs, mask=offs < X.shape[0])
    mean = tl.load(Mean + pid)
    rstd = tl.load(Rstd + pid)
    
    # Compute partial gradients
    dw = tl.sum(dy * (x - mean) * rstd, axis=0)
    db = tl.sum(dy, axis=0)
    
    # Use atomic operations to accumulate gradients
    tl.atomic_add(DW, dw)
    tl.atomic_add(DB, db)

# Wrapper class for LayerNorm
class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, B):
        BLOCK_SIZE = 128  # Example block size
        Y = torch.empty_like(X)
        Mean = torch.empty(X.size(0), device=X.device)
        Rstd = torch.empty(X.size(0), device=X.device)
        
        # Launch forward kernel
        _layer_norm_fwd_fused[(X.size(0) // BLOCK_SIZE,)](X, W, B, Y, Mean, Rstd, BLOCK_SIZE=BLOCK_SIZE)
        
        ctx.save_for_backward(X, W, Mean, Rstd)
        return Y

    @staticmethod
    def backward(ctx, DY):
        X, W, Mean, Rstd = ctx.saved_tensors
        BLOCK_SIZE = 128  # Example block size
        DX = torch.empty_like(X)
        DW = torch.zeros_like(W)
        DB = torch.zeros_like(B)
        
        # Launch backward kernels
        _layer_norm_bwd_dx_fused[(X.size(0) // BLOCK_SIZE,)](DY, X, W, Mean, Rstd, DX, BLOCK_SIZE=BLOCK_SIZE)
        _layer_norm_bwd_dwdb[(X.size(0) // BLOCK_SIZE,)](DY, X, Mean, Rstd, DW, DB, BLOCK_SIZE=BLOCK_SIZE)
        
        return DX, DW, DB

# Usage
class MyLayerNorm(torch.nn.Module):
    def __init__(self, num_features):
        super(MyLayerNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(num_features))
        self.bias = torch.nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return LayerNorm.apply(x, self.weight, self.bias)

# Example usage
x = torch.randn(64, 128, device='cuda')
ln = MyLayerNorm(128).cuda()
y = ln(x)
