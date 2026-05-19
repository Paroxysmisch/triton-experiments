import torch
import triton
import triton.language as tl

# Define the Triton kernel for RMS normalization
@triton.jit
def rms_norm_fwd_fused(x_ptr, w_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Pointers to data
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * n_elements
    w_row_ptr = w_ptr + row_idx * n_elements
    y_row_ptr = y_ptr + row_idx * n_elements

    # Load data for the row
    x = tl.load(x_row_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_elements, other=0.0)
    
    # Compute variance
    mean_square = tl.sum(x * x, axis=0) / n_elements
    inv_std = tl.rsqrt(mean_square + 1e-5)
    
    # Normalize and scale
    y = x * inv_std * tl.load(w_row_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_elements, other=1.0)
    
    # Store the result
    tl.store(y_row_ptr + tl.arange(0, BLOCK_SIZE), y, mask=tl.arange(0, BLOCK_SIZE) < n_elements)

# PyTorch module to wrap the Triton kernel
class TritonLlamaRMSNorm(torch.nn.Module):
    def __init__(self, feature_dim):
        super(TritonLlamaRMSNorm, self).__init__()
        assert feature_dim < 64 * 1024, "Feature dimension must be less than 64KB"
        self.feature_dim = feature_dim
        self.weight = torch.nn.Parameter(torch.ones(feature_dim))

    def forward(self, x):
        # Ensure input tensor is 2D
        assert x.ndim == 2 and x.shape[1] == self.feature_dim, "Input tensor must be of shape (batch_size, feature_dim)"
        
        # Allocate output tensor
        y = torch.empty_like(x)
        
        # Launch the Triton kernel
        grid = (x.shape[0],)
        rms_norm_fwd_fused[grid](
            x_ptr=x,
            w_ptr=self.weight,
            y_ptr=y,
            n_elements=self.feature_dim,
            BLOCK_SIZE=1024  # Adjust BLOCK_SIZE based on the GPU's capacity
        )
        
        return y

# Example usage
if __name__ == "__main__":
    batch_size = 128
    feature_dim = 1024  # Must be less than 64KB
    x = torch.randn(batch_size, feature_dim, device='cuda')

    rms_norm = TritonLlamaRMSNorm(feature_dim).cuda()
    y = rms_norm(x)
    print(y)
