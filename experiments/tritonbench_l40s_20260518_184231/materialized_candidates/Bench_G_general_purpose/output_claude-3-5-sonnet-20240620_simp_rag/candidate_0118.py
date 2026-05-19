import triton
import triton.language as tl
import torch
import math

@triton.jit
def logsumexp_fwd_kernel(
    X,  # Pointer to input tensor
    Z,  # Pointer to output tensor
    stride_x,  # Stride for input tensor
    stride_z,  # Stride for output tensor
    D,  # Size of reduction dimension
    B,  # Block size for parallel reduction
    scale,  # Optional scale factor
    BLOCK_SIZE: tl.constexpr,  # Number of elements to process per block
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute offsets
    offset_x = pid * stride_x
    offset_z = pid * stride_z
    
    # Create range for current block
    row_idx = tl.arange(0, BLOCK_SIZE)
    mask = row_idx < D
    
    # Load input values
    x_ptr = X + offset_x + row_idx
    x = tl.load(x_ptr, mask=mask, other=-float('inf'))
    
    # Apply optional scaling
    x = x * scale
    
    # Compute maximum for numerical stability
    x_max = tl.max(x, axis=0)
    
    # Compute exp(x - x_max) and sum
    x = tl.exp(x - x_max)
    x_sum = tl.sum(x, axis=0)
    
    # Compute final result: log(sum(exp(x))) = x_max + log(sum(exp(x - x_max)))
    output = x_max + tl.log(x_sum)
    
    # Store result
    tl.store(Z + offset_z, output)

def logsumexp_fwd(x, scale=1.0):
    """
    Compute log(sum(exp(x))) over the last dimension
    
    Args:
        x: Input tensor
        scale: Optional scale factor (default: 1.0)
    
    Returns:
        z: Output tensor with log-sum-exp computed over last dimension
    """
    # Handle input tensor shape
    *batch_dims, D = x.shape
    x_reshaped = x.reshape(-1, D)
    N = x_reshaped.shape[0]
    
    # Determine block size
    BLOCK_SIZE = triton.next_power_of_2(D)
    
    # Create output tensor
    z = torch.empty(N, device=x.device, dtype=x.dtype)
    
    # Configure grid and block sizes
    def grid(meta):
        return (N,)
    
    # Launch kernel
    logsumexp_fwd_kernel[grid](
        x_reshaped, z,
        x_reshaped.stride(0), z.stride(0),
        D, BLOCK_SIZE, scale,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    
    # Reshape output to match input batch dimensions
    return z.reshape(*batch_dims)

# Example usage
if __name__ == "__main__":
    # Create sample input
    x = torch.randn(2, 3, 4, device='cuda')
    
    # Compute log-sum-exp
    z = logsumexp_fwd(x)
    
    # Compare with PyTorch implementation
    z_torch = torch.logsumexp(x, dim=-1)
    
    # Check results
    print("Max difference:", torch.max(torch.abs(z - z_torch)))
