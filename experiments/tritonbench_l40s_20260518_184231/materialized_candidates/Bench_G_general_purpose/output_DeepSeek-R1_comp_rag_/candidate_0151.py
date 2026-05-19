import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_fwd_fused(
    X,    # Input tensor pointer
    Y,    # Output tensor pointer
    W,    # Weights pointer
    Rstd, # Reciprocal standard deviation pointer
    stride_x,  # Stride between rows in X
    N,     # Number of columns (features per row)
    eps,   # Numerical stability epsilon
    BLOCK_SIZE: tl.constexpr,  # Block size for processing
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_x
    cols = tl.arange(0, BLOCK_SIZE)
    
    # Compute sum of squares for variance calculation
    sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += x * x
    
    var = tl.sum(sum_sq) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Rstd + row_idx, rstd)
    
    # Apply normalization and weights
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(Y + row_start + cols, y, mask=mask)

class TritonLlamaRMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
    
    def forward(self, x):
        # Reshape input to 2D tensor (flatten batch dimensions)
        x_2d = x.contiguous().view(-1, x.size(-1))
        M, N = x_2d.shape
        
        # Determine optimal block size and validate
        max_fused = 65536 // x.element_size()  # 64KB limit
        block_size = min(max_fused, triton.next_power_of_2(N))
        if N > block_size:
            raise RuntimeError(f"Feature dimension {N} exceeds maximum block size {block_size}.")
        
        # Configure kernel launch parameters
        num_warps = min(max(block_size // 256, 1), 8)
        y = torch.empty_like(x)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        
        # Launch kernel grid with computed parameters
        grid = (M,)
        rms_norm_fwd_fused[grid](
            x_2d, y, self.weight, rstd,
            x_2d.stride(0), N, self.eps,
            BLOCK_SIZE=block_size,
            num_warps=num_warps
        )
        
        return y.view_as(x)  # Restore original shape
