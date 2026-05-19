import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['D'],
)
@triton.jit
def logsumexp_fwd_kernel(
    x_ptr,          # pointer to input tensor
    z_ptr,          # pointer to output tensor
    m_ptr,          # pointer to max values
    stride_xn,      # stride for batch dimension
    stride_xd,      # stride for feature dimension
    stride_zn,      # stride for output batch dimension
    N,              # batch size
    D,              # feature dimension size
    BLOCK_SIZE: tl.constexpr,  # size of parallel block
):
    # Program ID
    pid = tl.program_id(0)
    
    # Batch index
    batch_idx = pid
    
    # Compute memory offsets for this program instance
    x_offset = batch_idx * stride_xn
    z_offset = batch_idx * stride_zn
    
    # Initialize max value and accumulator
    max_val = float('-inf')
    
    # First pass: find max value
    for d in range(0, D, BLOCK_SIZE):
        # Create block mask for bounds checking
        block_mask = d + tl.arange(0, BLOCK_SIZE) < D
        
        # Load input values
        x_block_ptr = x_ptr + x_offset + d * stride_xd
        x_block = tl.load(x_block_ptr, mask=block_mask, other=float('-inf'))
        
        # Update max value
        max_val = tl.maximum(max_val, tl.max(x_block, axis=0))
    
    # Store max value
    tl.store(m_ptr + batch_idx, max_val)
    
    # Second pass: compute exp and sum
    acc = 0.0
    for d in range(0, D, BLOCK_SIZE):
        # Create block mask for bounds checking
        block_mask = d + tl.arange(0, BLOCK_SIZE) < D
        
        # Load input values
        x_block_ptr = x_ptr + x_offset + d * stride_xd
        x_block = tl.load(x_block_ptr, mask=block_mask, other=float('-inf'))
        
        # Compute exp(x - max_val) and accumulate
        acc += tl.sum(tl.exp(x_block - max_val) * block_mask, axis=0)
    
    # Compute final result: log(sum(exp(x - max_val))) + max_val
    result = tl.log(acc) + max_val
    
    # Store result
    tl.store(z_ptr + batch_idx, result)

def logsumexp_fwd(x: torch.Tensor, dtype=None) -> torch.Tensor:
    """
    Compute logsumexp over the last dimension of x.
    
    Args:
        x: Input tensor
        dtype: Optional output dtype
    
    Returns:
        Result of logsumexp operation
    """
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    elif len(x.shape) > 2:
        x = x.reshape(-1, x.shape[-1])
    
    # Get tensor dimensions
    N, D = x.shape
    
    # Determine block size based on D
    BLOCK_SIZE = min(triton.next_power_of_2(D), 1024)
    
    # Create output tensors
    z = torch.empty(N, device=x.device, dtype=x.dtype)
    m = torch.empty(N, device=x.device, dtype=x.dtype)
    
    # Launch kernel
    grid = (N,)
    logsumexp_fwd_kernel[grid](
        x_ptr=x.data_ptr(),
        z_ptr=z.data_ptr(),
        m_ptr=m.data_ptr(),
        stride_xn=D,
        stride_xd=1,
        stride_zn=1,
        N=N,
        D=D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Cast to desired dtype if specified
    if dtype is not None:
        z = z.to(dtype)
    
    return z
