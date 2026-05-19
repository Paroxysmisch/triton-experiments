import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(
    x_ptr,          # Pointer to input tensor [B, M, K]
    rms_weights_ptr,# Pointer to weight tensor [K]
    out_ptr,        # Pointer to output tensor [B, M, K]
    batch_size,     # Batch dimension size
    seq_len,        # Sequence length dimension size
    hidden_dim,     # Hidden dimension size (K)
    stride_b,       # Stride for batch dimension
    stride_m,       # Stride for sequence dimension
    stride_k,       # Stride for hidden dimension
    BLOCK_SIZE: tl.constexpr,  # Block size for K dimension
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate batch and sequence indices
    batch_id = pid // seq_len
    seq_id = pid % seq_len
    
    # Compute the starting offset for this batch,seq position
    offset_bm = batch_id * stride_b + seq_id * stride_m
    
    # Load K elements for this batch,seq position
    ptr = x_ptr + offset_bm
    
    # Initialize accumulator for mean square
    rms = 0.0
    
    # Load and square elements
    for k in range(0, hidden_dim, BLOCK_SIZE):
        mask = k + tl.arange(0, BLOCK_SIZE) < hidden_dim
        x = tl.load(ptr + k * stride_k, mask=mask, other=0.0)
        rms += tl.sum(x * x, mask=mask)
    
    # Compute RMS
    rms = tl.sqrt(rms / hidden_dim + 1e-6)
    
    # Normalize and apply weights
    for k in range(0, hidden_dim, BLOCK_SIZE):
        mask = k + tl.arange(0, BLOCK_SIZE) < hidden_dim
        x = tl.load(ptr + k * stride_k, mask=mask, other=0.0)
        w = tl.load(rms_weights_ptr + k, mask=mask, other=0.0)
        
        out = (x / rms) * w
        tl.store(out_ptr + offset_bm + k * stride_k, out, mask=mask)

def rmsnorm_wrapper(x: torch.Tensor, rms_weights: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for RMSNorm Triton kernel
    Args:
        x: Input tensor of shape [B, M, K]
        rms_weights: Weight tensor of shape [K]
    Returns:
        Normalized tensor of shape [B, M, K]
    """
    assert x.dim() == 3, "Input tensor must be 3D"
    assert rms_weights.dim() == 1, "Weight tensor must be 1D"
    assert x.shape[2] == rms_weights.shape[0], "Hidden dimension mismatch"
    
    batch_size, seq_len, hidden_dim = x.shape
    
    # Ensure tensors are contiguous and in correct format
    x = x.contiguous()
    rms_weights = rms_weights.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Calculate strides
    stride_b = x.stride(0)
    stride_m = x.stride(1)
    stride_k = x.stride(2)
    
    # Configure kernel parameters
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)  # Maximum block size
    
    # Launch kernel
    grid = (batch_size * seq_len,)
    rmsnorm_triton[(grid,)](
        x_ptr=x,
        rms_weights_ptr=rms_weights,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        stride_b=stride_b,
        stride_m=stride_m,
        stride_k=stride_k,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
