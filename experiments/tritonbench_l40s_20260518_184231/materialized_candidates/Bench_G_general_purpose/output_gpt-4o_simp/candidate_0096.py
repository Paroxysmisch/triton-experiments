import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_fwd_fused(x_ptr, weight_ptr, y_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    # Program ID corresponds to the row index
    row_idx = tl.program_id(0)
    
    # Offsets for the row
    offsets = row_idx * n_cols + tl.arange(0, BLOCK_SIZE)
    
    # Load input data for the current row
    x = tl.load(x_ptr + offsets, mask=offsets < n_cols, other=0.0)
    
    # Compute the squared values
    x_squared = x * x
    
    # Compute the sum of squares for the row
    sum_squares = tl.sum(x_squared, axis=0)
    
    # Compute the mean square
    mean_square = sum_squares / n_cols
    
    # Compute the inverse of the RMS (root mean square)
    inv_rms = 1.0 / tl.sqrt(mean_square + eps)
    
    # Load the weight vector
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=0.0)
    
    # Normalize and apply the weight
    y = x * inv_rms * weight
    
    # Store the result
    tl.store(y_ptr + offsets, y, mask=offsets < n_cols)

def rmsnorm_forward(x, weight, eps=1e-5):
    # Get the shape of the input tensor
    n_rows, n_cols = x.shape
    
    # Allocate output tensor
    y = torch.empty_like(x)
    
    # Launch Triton kernel
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    
    _rms_norm_fwd_fused[grid](
        x_ptr=x,
        weight_ptr=weight,
        y_ptr=y,
        n_cols=n_cols,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y

# Example usage
x = torch.randn(1024, 512, device='cuda')
weight = torch.ones(512, device='cuda')  # Example weight vector
y = rmsnorm_forward(x, weight)
