import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_fwd_fused(
    X,  # pointer to input
    Y,  # pointer to output
    W,  # pointer to weights
    stride,  # stride for row offset
    N,  # number of columns
    eps,  # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # compile-time constant for parallel processing
):
    # Get row index from program ID
    row = tl.program_id(0)
    
    # Offset pointers to current row
    Y += row * stride
    X += row * stride
    
    # Initialize variance accumulator
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Compute variance in blocks
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _var += x * x
    
    # Calculate RMS normalization factor
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    
    # Apply normalization and weights
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(Y + cols, y, mask=mask)

class TritonLlamaRMSNorm(torch.nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = weight
        self.eps = eps
    
    def forward(self, x):
        # Prepare output tensor
        y = torch.empty_like(x)
        
        # Reshape input to 2D
        x_reshaped = x.reshape(-1, x.shape[-1])
        M, N = x_reshaped.shape
        
        # Calculate optimal block size
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        
        # Check dimension constraints
        if N > BLOCK_SIZE:
            raise RuntimeError("Feature dimension must be < 64KB")
        
        # Calculate optimal number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        
        # Launch kernel
        rms_norm_fwd_fused[(M,)](
            x_reshaped,
            y,
            self.weight,
            x_reshaped.stride(0),
            N,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        
        return y
