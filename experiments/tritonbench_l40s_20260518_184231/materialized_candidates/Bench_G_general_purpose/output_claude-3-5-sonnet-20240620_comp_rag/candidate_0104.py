import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_fwd_fused(
    X,      # pointer to input tensor
    Y,      # pointer to output tensor
    W,      # pointer to weight tensor
    stride, # stride for moving between rows
    N,      # number of columns
    eps,    # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # size of parallel processing block
):
    # Get the row index this program instance should process
    row = tl.program_id(0)
    
    # Offset pointers to the correct row
    Y += row * stride
    X += row * stride
    
    # Initialize accumulator for variance calculation
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # First pass: compute variance
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        # Load input values with masking for boundary conditions
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        # Square and accumulate for variance
        _var += x * x
    
    # Compute RMS statistics
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    
    # Second pass: normalize and scale
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        # Load input and weights
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        # Normalize and scale
        x_hat = x * rstd
        y = x_hat * w
        # Store result
        tl.store(Y + cols, y, mask=mask)

class TritonLlamaRMSNorm(torch.nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, x):
        # Prepare output tensor
        y = torch.empty_like(x)
        
        # Reshape input to 2D for processing
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        
        # Calculate optimal block size
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        
        # Check feature dimension constraint
        if N > BLOCK_SIZE:
            raise RuntimeError("Feature dimension must be < 64KB")
        
        # Calculate optimal number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        
        # Launch kernel
        rms_norm_fwd_fused[(M,)](
            x_arg,
            y,
            self.weight,
            x_arg.stride(0),
            N,
            self.variance_epsilon,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        
        return y
