import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def rms_norm_fwd_kernel(
    X,  # Pointer to input tensor
    Y,  # Pointer to output tensor
    W,  # Pointer to weight tensor
    Rstd,  # Pointer to reciprocal standard deviation
    stride,  # Stride for moving between rows
    N,  # Number of columns
    eps,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # Static block size for optimization
):
    # Get the row index
    row = tl.program_id(0)
    
    # Offset pointers to current row
    X_row = X + row * stride
    Y_row = Y + row * stride
    
    # Compute variance
    var = tl.zeros([1], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        var += tl.sum(x * x * mask, axis=0)
    
    var = var / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    
    # Normalize and scale
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        
        y = x * rstd * w
        tl.store(Y_row + cols, y, mask=mask)

class TritonRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.dim = dim
        
        # Compute optimal block size
        self.BLOCK_SIZE = min(
            triton.next_power_of_2(dim),
            1024  # Maximum block size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous input
        x = x.contiguous()
        
        # Reshape input if needed
        orig_shape = x.shape
        x = x.view(-1, self.dim)
        
        # Allocate output
        y = torch.empty_like(x)
        rstd = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        
        # Launch kernel
        grid = (x.shape[0],)
        rms_norm_fwd_kernel[grid](
            x,
            y,
            self.weight,
            rstd,
            x.stride(0),
            self.dim,
            self.eps,
            BLOCK_SIZE=self.BLOCK_SIZE,
        )
        
        return y.view(orig_shape)
