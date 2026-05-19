import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    X,  # Input tensor pointer
    Y,  # Output tensor pointer
    W,  # Weight tensor pointer
    stride,  # Row stride
    N,  # Row length
    eps,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # Number of elements to process in parallel
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Compute the row offset
    row_start = pid * stride
    
    # Create offsets for this program instance
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Load data for this row
    x = tl.load(X + row_start + offs, mask=mask, other=0.0)
    w = tl.load(W + offs, mask=mask, other=1.0)
    
    # Compute variance
    x2 = x * x
    var = tl.sum(x2, axis=0) / N
    
    # Compute RMS normalization factor
    rrms = 1.0 / tl.sqrt(var + eps)
    
    # Apply normalization and scaling
    y = (x * rrms).to(Y.dtype.element_ty) * w
    
    # Store the result
    tl.store(Y + row_start + offs, y, mask=mask)

class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        # Save context for backward pass
        ctx.eps = eps
        ctx.N = x.shape[-1]
        
        # Allocate output
        y = torch.empty_like(x)
        
        # Configure grid and block sizes
        BLOCK_SIZE = triton.next_power_of_2(ctx.N)
        grid = (x.shape[0],)  # One block per row
        
        # Launch kernel
        rms_norm_kernel[grid](
            x,
            y,
            weight,
            x.stride(0),
            ctx.N,
            eps,
            BLOCK_SIZE,
        )
        return y

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Apply RMS normalization to the input tensor.
    
    Args:
        x: Input tensor to normalize
        weight: Scaling weights
        eps: Small constant for numerical stability
    
    Returns:
        RMS normalized tensor
    """
    return RmsNorm.apply(x, weight, eps)
