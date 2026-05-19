import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_fwd_fused(
    X,  # pointer to input tensor
    Y,  # pointer to output tensor
    W,  # pointer to weight tensor
    stride,  # stride between rows
    N,  # number of columns
    eps,  # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # block size for parallel processing
):
    # Get the row index
    row = tl.program_id(0)
    
    # Compute pointers for this row
    X_row_ptr = X + row * stride
    Y_row_ptr = Y + row * stride

    # Initialize accumulator for variance calculation
    var = tl.zeros([1], dtype=tl.float32)
    
    # Step 1: Compute variance
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        var += tl.sum(x * x, axis=0)
    
    # Finalize variance computation
    var = var / N
    
    # Compute reciprocal of standard deviation
    rstd = 1 / tl.sqrt(var + eps)
    
    # Step 2: Normalize and scale
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        # Load values
        x = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        
        # Normalize and apply weight
        y = x * rstd * w
        
        # Store result
        tl.store(Y_row_ptr + cols, y, mask=mask)

class TritonLlamaRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        # Ensure input is contiguous and get dimensions
        x = x.contiguous()
        orig_shape = x.shape
        x = x.view(-1, x.size(-1))
        M, N = x.shape
        
        # Allocate output tensor
        y = torch.empty_like(x)
        
        # Calculate optimal block size (must be power of 2)
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        
        # Ensure N is not too large
        if N > BLOCK_SIZE:
            raise RuntimeError("Feature dimension must be < 64KB")
        
        # Launch kernel
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        rms_norm_fwd_fused[(M,)](
            x, y, self.weight,
            x.stride(0), N, self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        
        # Restore original shape
        return y.view(orig_shape)
