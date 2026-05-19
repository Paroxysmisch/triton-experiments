import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    # Pointers to matrices
    x_ptr,          # Input tensor pointer
    w_ptr,          # Weight tensor pointer
    output_ptr,     # Output tensor pointer
    # Matrix dimensions
    batch_stride,   # Stride for batch dimension
    row_stride,     # Stride for row dimension
    col_stride,     # Stride for column dimension
    weight_stride,  # Stride for weight tensor
    # Constants
    seq_len: tl.constexpr,  # Sequence length (K dimension)
    eps: tl.constexpr,      # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr, # Block size for parallel processing
):
    # Get program ID for batch and row dimensions
    batch_pid = tl.program_id(0)
    row_pid = tl.program_id(1)
    
    # Calculate row offset
    row_offset = batch_pid * batch_stride + row_pid * row_stride
    
    # Create block offset range
    offs = tl.arange(0, BLOCK_SIZE)
    
    # Initialize variance accumulator
    var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # First pass: compute variance
    for block_start in range(0, seq_len, BLOCK_SIZE):
        col_offs = block_start + offs
        mask = col_offs < seq_len
        
        # Load and square values
        x = tl.load(x_ptr + row_offset + col_offs * col_stride, mask=mask, other=0.0)
        x = x.to(tl.float32)
        var += x * x
    
    # Compute RMS statistics
    var = tl.sum(var) / seq_len
    rms = tl.sqrt(var + eps)
    
    # Second pass: normalize and apply weights
    for block_start in range(0, seq_len, BLOCK_SIZE):
        col_offs = block_start + offs
        mask = col_offs < seq_len
        
        # Load input and weights
        x = tl.load(x_ptr + row_offset + col_offs * col_stride, mask=mask, other=0.0)
        w = tl.load(w_ptr + col_offs * weight_stride, mask=mask, other=0.0)
        
        # Normalize and scale
        x = x.to(tl.float32)
        output = (x / rms) * w
        
        # Store result
        tl.store(output_ptr + row_offset + col_offs * col_stride, output, mask=mask)

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Apply RMS normalization to input tensor.
    
    Args:
        x: Input tensor of shape [batch_size, seq_len, hidden_dim]
        weight: Weight tensor of shape [hidden_dim]
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor of same shape as input
    """
    batch_size, seq_len, hidden_dim = x.shape
    output = torch.empty_like(x)
    
    # Launch kernel with appropriate grid
    grid = (batch_size, seq_len)
    rms_norm_kernel[grid](
        x_ptr=x,
        w_ptr=weight,
        output_ptr=output,
        batch_stride=x.stride(0),
        row_stride=x.stride(1),
        col_stride=x.stride(2),
        weight_stride=weight.stride(0),
        seq_len=hidden_dim,
        eps=eps,
        BLOCK_SIZE=min(hidden_dim, 4096),  # Choose optimal block size
        num_warps=4
    )
    
    return output
