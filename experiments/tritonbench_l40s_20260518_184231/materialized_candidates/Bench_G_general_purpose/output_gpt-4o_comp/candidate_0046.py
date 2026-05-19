import triton
import triton.language as tl
import torch

# Forward kernel for L2 normalization
@triton.jit
def _l2_norm_fwd_1pass_kernel(X, Y, stride_x, stride_y, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = row * stride_x + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets, mask=offsets < X.shape[1], other=0.0)
    
    # Compute sum of squares
    sq = x * x
    sum_sq = tl.sum(sq, axis=0)
    
    # Compute reciprocal of square root of sum of squares
    rstd = tl.rsqrt(sum_sq)
    
    # Normalize
    y = x * rstd
    tl.store(Y + offsets, y, mask=offsets < Y.shape[1])

# Backward kernel for L2 normalization
@triton.jit
def _l2_norm_bwd_kernel(X, DY, DX, stride_x, stride_dy, stride_dx, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = row * stride_x + tl.arange(0, BLOCK_SIZE)
    
    x = tl.load(X + offsets, mask=offsets < X.shape[1], other=0.0)
    dy = tl.load(DY + offsets, mask=offsets < DY.shape[1], other=0.0)
    
    # Compute sum of squares
    sq = x * x
    sum_sq = tl.sum(sq, axis=0)
    
    # Compute reciprocal of square root of sum of squares
    rstd = tl.rsqrt(sum_sq)
    
    # Compute gradient
    dx = dy * rstd - x * tl.sum(dy * x, axis=0) * rstd * rstd * rstd
    tl.store(DX + offsets, dx, mask=offsets < DX.shape[1])

# Wrapper for forward pass
def _l2_norm_fwd(X):
    assert X.stride(-1) == 1, "Last dimension stride must be 1."
    BLOCK_SIZE = 1024  # This should be tuned based on the hardware and problem size
    Y = torch.empty_like(X)
    
    grid = (X.shape[0],)
    _l2_norm_fwd_1pass_kernel[grid](X, Y, X.stride(0), Y.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    
    return Y

# Wrapper for backward pass
def _l2_norm_bwd(X, DY):
    assert X.stride(-1) == 1, "Last dimension stride must be 1."
    BLOCK_SIZE = 1024  # This should be tuned based on the hardware and problem size
    DX = torch.empty_like(X)
    
    grid = (X.shape[0],)
    _l2_norm_bwd_kernel[grid](X, DY, DX, X.stride(0), DY.stride(0), DX.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    
    return DX

# Example usage
X = torch.randn(128, 1024, device='cuda')
DY = torch.randn(128, 1024, device='cuda')

# Forward pass
Y = _l2_norm_fwd(X)

# Backward pass
DX = _l2_norm_bwd(X, DY)
