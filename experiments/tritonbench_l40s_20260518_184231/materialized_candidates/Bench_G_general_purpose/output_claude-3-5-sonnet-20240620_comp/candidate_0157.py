import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_fwd_fused(
    X_ptr,  # pointer to input tensor
    Y_ptr,  # pointer to output tensor
    W_ptr,  # pointer to weight tensor
    stride, # row stride
    N,      # number of columns
    eps,    # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    row_idx = tl.program_id(0)
    
    # Compute pointers for current row
    row_start_ptr = X_ptr + row_idx * stride
    
    # Initialize accumulator for variance calculation
    acc = 0.0
    
    # Load and square elements in blocks
    for block_start in range(0, N, BLOCK_SIZE):
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols + block_start < N
        x = tl.load(row_start_ptr + block_start + cols, mask=mask, other=0.0)
        acc += tl.sum(x * x * mask, axis=0)
    
    # Compute RMS statistics
    rms = tl.sqrt(acc / N + eps)
    rstd = 1.0 / rms
    
    # Apply normalization with weights
    row_out_ptr = Y_ptr + row_idx * stride
    for block_start in range(0, N, BLOCK_SIZE):
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols + block_start < N
        
        # Load input and weights
        x = tl.load(row_start_ptr + block_start + cols, mask=mask, other=0.0)
        w = tl.load(W_ptr + block_start + cols, mask=mask, other=1.0)
        
        # Normalize and scale
        y = x * rstd * w
        
        # Store result
        tl.store(row_out_ptr + block_start + cols, y, mask=mask)

class TritonLlamaRMSNorm(torch.nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, x):
        # Reshape input to 2D if needed
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        
        # Prepare output tensor
        y = torch.empty_like(x)
        
        # Calculate optimal block size (power of 2)
        N = x.shape[1]
        BLOCK_SIZE = min(triton.next_power_of_2(N), 1024)
        
        # Launch kernel
        grid = (x.shape[0],)  # One thread block per row
        rms_norm_fwd_fused[grid](
            x.data_ptr(),
            y.data_ptr(),
            self.weight.data_ptr(),
            x.stride(0),
            N,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Restore original shape
        return y.reshape(orig_shape)

# Example usage
def test_rms_norm():
    B, H = 4, 256  # batch size and hidden dimension
    x = torch.randn(B, H, device='cuda')
    weight = torch.ones(H, device='cuda')
    
    # Create and run RMS norm
    rms_norm = TritonLlamaRMSNorm(weight)
    y = rms_norm(x)
    return y
