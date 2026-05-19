import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_triton(
    x_ptr,          # pointer to input tensor [batch, M, K]
    rms_w_ptr,      # pointer to RMS weights [K]
    out_ptr,        # pointer to output tensor [batch, M, K]
    stride_b,       # stride for batch dimension
    stride_m,       # stride for M dimension
    stride_k,       # stride for K dimension
    N_SIZE,         # size of K dimension
    BLOCK_N_SIZE,   # block size for K dimension processing
    batch,          # batch size
    M,              # M dimension size
    eps: tl.float32 # epsilon for numerical stability
):
    # Get program ID for batch and M dimensions
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    # Bounds checking
    if pid_batch >= batch or pid_m >= M:
        return
        
    # Compute base offset for current batch and M position
    base_offset = pid_batch * stride_b + pid_m * stride_m
    
    # Create offsets for K dimension
    offs_k = tl.arange(0, BLOCK_N_SIZE)
    mask_k = offs_k < N_SIZE
    
    # Initialize accumulator for sum of squares
    square_sum = tl.zeros([1], dtype=tl.float32)
    
    # First pass: compute sum of squares
    for k in range(0, N_SIZE, BLOCK_N_SIZE):
        x_offs = base_offset + (k + offs_k) * stride_k
        x_block = tl.load(x_ptr + x_offs, mask=mask_k & (k + offs_k < N_SIZE))
        square_sum += tl.sum(x_block * x_block * mask_k, axis=0)
    
    # Compute RMS
    rms = tl.sqrt(square_sum / N_SIZE + eps)
    
    # Second pass: normalize and scale
    for k in range(0, N_SIZE, BLOCK_N_SIZE):
        k_offs = k + offs_k
        mask = mask_k & (k_offs < N_SIZE)
        
        # Load input and weights
        x_offs = base_offset + k_offs * stride_k
        x_block = tl.load(x_ptr + x_offs, mask=mask)
        w_block = tl.load(rms_w_ptr + k_offs, mask=mask)
        
        # Normalize and scale
        out_block = (x_block / rms) * w_block
        
        # Store result
        tl.store(out_ptr + x_offs, out_block, mask=mask)

def rmsnorm_wrapper(x: torch.Tensor, rms_weights: torch.Tensor, eps: float = 1e-6):
    """
    Wrapper function for RMSNorm Triton kernel
    Args:
        x: Input tensor of shape [batch, M, K]
        rms_weights: Weights tensor of shape [K]
        eps: Small constant for numerical stability
    Returns:
        Normalized tensor of same shape as input
    """
    assert x.dim() == 3, "Input tensor must be 3D"
    batch, M, K = x.shape
    
    # Ensure tensors are contiguous and in correct format
    x = x.contiguous()
    rms_weights = rms_weights.contiguous()
    
    # Compute strides
    stride_b = M * K
    stride_m = K
    stride_k = 1
    
    # Determine block size (power of 2 <= K)
    BLOCK_N_SIZE = min(triton.next_power_of_2(K), 512)
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Configure grid and block sizes
    grid = (batch, M)
    num_warps = 4 if K <= 512 else 8
    
    # Launch kernel
    rmsnorm_triton[grid](
        x_ptr=x,
        rms_w_ptr=rms_weights,
        out_ptr=output,
        stride_b=stride_b,
        stride_m=stride_m,
        stride_k=stride_k,
        N_SIZE=K,
        BLOCK_N_SIZE=BLOCK_N_SIZE,
        batch=batch,
        M=M,
        eps=eps,
        num_warps=num_warps
    )
    
    return output
