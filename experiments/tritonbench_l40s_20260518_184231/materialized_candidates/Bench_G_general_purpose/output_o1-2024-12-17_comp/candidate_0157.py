import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def rms_norm_fwd_fused(
    X, 
    Y, 
    W, 
    stride_x, 
    stride_y, 
    stride_w, 
    N, 
    eps, 
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    row_offset_x = row_id * stride_x
    row_offset_y = row_id * stride_y
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(X + row_offset_x + offsets, mask=mask, other=0.0)
    sq = x * x
    vsum = tl.sum(sq, axis=0)
    vsum = vsum / N
    vsum = vsum + eps
    rstd = 1.0 / tl.sqrt(vsum)
    
    w = tl.load(W + offsets, mask=mask, other=0.0)
    
    out = x * rstd * w
    tl.store(Y + row_offset_y + offsets, out, mask=mask)

class TritonLlamaRMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, "Input tensor must be 2D."
        
        M, N = x.shape
        y = torch.empty_like(x)
        
        block_size = 1 << int(math.log2(N))
        block_size = min(block_size, 1024)
        
        grid = (M,)
        
        rms_norm_fwd_fused[grid](
            x,
            y,
            self.weight,
            x.stride(0),
            y.stride(0),
            self.weight.stride(0),
            N,
            self.eps,
            BLOCK_SIZE=block_size
        )
        
        return y
