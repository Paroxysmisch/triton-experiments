import triton
import triton.language as tl

@triton.jit
def _rms_norm_fwd_fused(
    x_ptr,
    weight_ptr,
    y_ptr,
    stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the row index
    row_idx = tl.program_id(0)
    
    # Compute pointers to the row
    x_row_ptr = x_ptr + row_idx * stride
    y_row_ptr = y_ptr + row_idx * stride
    
    # Load the row into SRAM
    x_row = tl.load(x_row_ptr + tl.arange(0, BLOCK_SIZE))
    
    # Compute variance
    var = tl.sum(x_row * x_row) / n_cols
    
    # Compute normalization factor
    norm_factor = 1 / tl.sqrt(var + eps)
    
    # Load weights
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE))
    
    # Normalize and scale
    y_row = x_row * norm_factor * weight
    
    # Store the result
    tl.store(y_row_ptr + tl.arange(0, BLOCK_SIZE), y_row)

# Wrapper function
def rmsnorm_forward(x, weight, eps=1e-6):
    # Get input shape
    batch_size, seq_len, hidden_dim = x.shape
    
    # Reshape input to 2D
    x_2d = x.view(-1, hidden_dim)
    
    # Allocate output
    y = torch.empty_like(x_2d)
    
    # Define grid and block sizes
    grid = (x_2d.shape[0],)
    block_size = triton.next_power_of_2(hidden_dim)
    block_size = min(block_size, 1024)  # Ensure block size doesn't exceed 1024
    
    # Launch kernel
    _rms_norm_fwd_fused[grid](
        x_2d,
        weight,
        y,
        x_2d.stride(0),
        hidden_dim,
        eps,
        BLOCK_SIZE=block_size
    )
    
    # Reshape output back to 3D
    return y.view(batch_size, seq_len, hidden_dim)
