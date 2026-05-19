import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_fwd_fused(
    X,  # Input tensor pointer
    Y,  # Output tensor pointer
    W,  # Weight tensor pointer
    stride_x,  # Stride between rows in X/Y
    N,  # Number of columns
    eps,  # Numerical stability term
    BLOCK_SIZE: tl.constexpr,  # Block size for computation
):
    row = tl.program_id(0)  # Row index in 2D tensor
    X_row = X + row * stride_x  # Compute starting pointers
    Y_row = Y + row * stride_x  # for current row

    # Variance calculation loop
    var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        var += x * x  # Accumulate squared values

    var = tl.sum(var, axis=0) / N  # Calculate mean of squares
    rstd = 1 / tl.sqrt(var + eps)  # Reciprocal standard deviation

    # Normalization and weight application loop
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)  # Load weights
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd  # Normalize
        y = x_hat * w  # Apply weights
        tl.store(Y_row + cols, y, mask=mask)  # Store results


class TritonLlamaRMSNorm(torch.nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', torch.nn.Parameter(weight))
        self.eps = eps

    def forward(self, x):
        B, T, C = x.shape  # Handle 3D tensors
        x_2d = x.reshape(-1, C)  # Flatten batch dimensions
        M, N = x_2d.shape
        
        # Optimize block size and warps
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if BLOCK_SIZE < N:
            raise ValueError(f"Feature dimension {N} exceeds supported {BLOCK_SIZE}")
        
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        
        # Allocate output and launch kernel
        y = torch.empty_like(x_2d)
        rms_norm_fwd_fused[(M,)](  # Grid of M rows
            x_2d, y, self.weight,
            x_2d.stride(0), N, self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        return y.reshape_as(x)  # Restore original shape
