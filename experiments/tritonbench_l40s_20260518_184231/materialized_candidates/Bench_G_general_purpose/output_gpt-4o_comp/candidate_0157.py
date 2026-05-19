import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_fwd_fused(X, Y, W, stride_x, stride_y, stride_w, N, eps, BLOCK_SIZE: tl.constexpr):
    # Program IDs
    pid = tl.program_id(0)
    
    # Block start and offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data from X
    x_ptrs = X + offsets[:, None] * stride_x + tl.arange(0, N)[None, :]
    x = tl.load(x_ptrs, mask=offsets[:, None] < N, other=0.0)
    
    # Compute mean square
    mean_square = tl.sum(x * x, axis=1) / N
    
    # Compute rstd (1 / sqrt(variance + eps))
    rstd = 1.0 / tl.sqrt(mean_square + eps)
    
    # Normalize and apply weight
    w_ptrs = W + offsets[:, None] * stride_w
    w = tl.load(w_ptrs, mask=offsets[:, None] < N, other=0.0)
    y = (x * rstd[:, None]) * w
    
    # Store result
    y_ptrs = Y + offsets[:, None] * stride_y + tl.arange(0, N)[None, :]
    tl.store(y_ptrs, y, mask=offsets[:, None] < N)

class TritonLlamaRMSNorm(torch.nn.Module):
    def __init__(self, weight, eps=1e-5):
        super(TritonLlamaRMSNorm, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, x):
        # Reshape input tensor to 2D
        batch_size, num_features = x.shape
        x_2d = x.view(batch_size, num_features)
        
        # Output tensor
        y = torch.empty_like(x_2d)
        
        # Define block size
        BLOCK_SIZE = 128  # This can be tuned based on your GPU architecture
        
        # Enqueue Triton kernel
        rms_norm_fwd_fused[(batch_size,)](
            x_2d, y, self.weight,
            x_2d.stride(0), y.stride(0), self.weight.stride(0),
            num_features, self.eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return y.view_as(x)

# Example usage
if __name__ == "__main__":
    # Sample data
    x = torch.randn(32, 512, device='cuda')  # Example input
    weight = torch.ones(512, device='cuda')  # Example weight

    # Create RMSNorm layer
    rms_norm = TritonLlamaRMSNorm(weight)
    
    # Forward pass
    y = rms_norm(x)
    print(y)
