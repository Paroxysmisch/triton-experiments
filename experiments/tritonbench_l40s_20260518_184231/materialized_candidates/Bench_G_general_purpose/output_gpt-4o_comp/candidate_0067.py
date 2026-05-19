import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(X, W, Y, N, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Compute the start index for the current row
    row_start = pid * BLOCK_SIZE
    
    # Create pointers for the row of X, W, and Y
    x_ptrs = X + row_start * N + tl.arange(0, BLOCK_SIZE)
    w_ptrs = W + tl.arange(0, BLOCK_SIZE)
    y_ptrs = Y + row_start * N + tl.arange(0, BLOCK_SIZE)
    
    # Load the current row of X and W
    x = tl.load(x_ptrs, mask=tl.arange(0, BLOCK_SIZE) < N)
    w = tl.load(w_ptrs, mask=tl.arange(0, BLOCK_SIZE) < N)
    
    # Compute the variance
    var = tl.sum(x * x, axis=0) / N
    
    # Compute the reciprocal of the root mean square
    rrms = 1.0 / tl.sqrt(var + eps)
    
    # Normalize the row and apply weights
    y = (x * rrms).to(Y.dtype.element_ty) * w
    
    # Store the result in Y
    tl.store(y_ptrs, y, mask=tl.arange(0, BLOCK_SIZE) < N)

class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-5):
        assert x.is_contiguous() and weight.is_contiguous(), "Inputs must be contiguous"
        B, N = x.shape
        y = torch.empty_like(x)
        
        # Launch the Triton kernel
        grid = (B,)
        rms_norm_kernel[grid](x, weight, y, N, eps, BLOCK_SIZE=N)
        
        return y

def rms_norm(x, normalized_shape, weight, eps=1e-5):
    return RmsNorm.apply(x, weight, eps)

# Example usage:
B, N = 32, 128  # Batch size and feature size
x = torch.randn(B, N, device='cuda', dtype=torch.float32)
weight = torch.ones(N, device='cuda', dtype=torch.float32)

# Perform RMS normalization
y = rms_norm(x, (N,), weight)
