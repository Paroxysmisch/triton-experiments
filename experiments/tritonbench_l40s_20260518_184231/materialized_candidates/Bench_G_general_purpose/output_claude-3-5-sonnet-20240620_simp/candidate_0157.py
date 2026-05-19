import torch
import triton
import triton.language as tl
import math

@triton.jit
def rms_norm_fwd_fused(
    X,  # pointer to input tensor
    Y,  # pointer to output tensor
    W,  # pointer to weights
    stride,  # stride between rows
    N,  # number of columns
    BLOCK_SIZE: tl.constexpr,
):
    # Map program ID to the row of X and Y it should compute
    row_idx = tl.program_id(0)
    
    # Compute pointers to current row
    x_ptr = X + row_idx * stride
    y_ptr = Y + row_idx * stride
    
    # Load data to SRAM
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)
    
    # Compute variance
    x2 = x * x
    var = tl.sum(x2, axis=0) / N
    
    # Compute RMS normalization
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    
    # Apply normalization and weight
    y = x * rstd * w
    
    # Write back result
    tl.store(y_ptr + cols, y, mask=mask)

class TritonLlamaRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.dim = dim
        
        # Compute block size (must be power of 2)
        self.BLOCK_SIZE = triton.next_power_of_2(dim)
        assert self.BLOCK_SIZE <= 65536, "Feature dimension must be <= 64KB"
        
    def forward(self, x):
        # Reshape input if needed
        orig_shape = x.shape
        x = x.view(-1, self.dim)
        
        # Allocate output
        y = torch.empty_like(x)
        
        # Enqueue kernel
        grid = (x.shape[0],)  # Number of rows
        rms_norm_fwd_fused[grid](
            x, y, self.weight,
            x.stride(0),
            self.dim,
            self.BLOCK_SIZE,
        )
        
        # Restore original shape
        y = y.view(orig_shape)
        return y

# Example usage
if __name__ == "__main__":
    # Create sample input
    batch_size = 32
    seq_len = 128
    dim = 512
    x = torch.randn(batch_size, seq_len, dim, device='cuda')
    
    # Initialize layer
    rms_norm = TritonLlamaRMSNorm(dim).cuda()
    
    # Forward pass
    y = rms_norm(x)
