import triton
import triton.language as tl
import torch

@triton.jit
def triton_red_fused_native_layer_norm_0(
    primals_3_ptr, # input tensor pointer
    primals_1_ptr, # scale pointer
    primals_2_ptr, # bias pointer
    out_ptr0,      # buffer for mean, var, count
    out_ptr1,      # output tensor pointer
    stride_s3_0,   # stride for dim 0
    stride_s3_1,   # stride for dim 1
    stride_s1_0,   # stride for scale
    stride_s2_0,   # stride for bias
    stride_out_0,  # output stride dim 0
    stride_out_1,  # output stride dim 1
    S,             # sequence length
    D,             # hidden dimension
    RBLOCK: tl.constexpr,  # reduction block size
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Initialize shared memory for reduction
    tmp3_mean = tl.zeros([RBLOCK], dtype=tl.float32)
    tmp3_m2 = tl.zeros([RBLOCK], dtype=tl.float32)
    tmp3_weight = tl.zeros([RBLOCK], dtype=tl.float32)
    
    # Compute reduction ranges
    row_idx = pid
    col_start = tl.arange(0, RBLOCK)
    
    # Load and reduce
    for col in range(0, D, RBLOCK):
        cols = col + col_start
        mask = cols < D
        
        # Load values
        x = tl.load(primals_3_ptr + row_idx * stride_s3_0 + cols * stride_s3_1, mask=mask, other=0.0)
        
        # Welford update
        weight = mask
        delta = x - tmp3_mean
        tmp3_mean += delta * weight / (tmp3_weight + weight)
        delta2 = x - tmp3_mean
        tmp3_m2 += delta * delta2 * weight
        tmp3_weight += weight
    
    # Store reduction results
    tl.store(out_ptr0 + pid * 3 + 0, tmp3_mean[0])  # mean
    tl.store(out_ptr0 + pid * 3 + 1, tmp3_m2[0])    # variance
    tl.store(out_ptr0 + pid * 3 + 2, tmp3_weight[0]) # count
    
    # Sync threads
    tl.debug_barrier()
    
    # Load statistics
    mean = tl.load(out_ptr0 + pid * 3 + 0)
    var = tl.load(out_ptr0 + pid * 3 + 1)
    count = tl.load(out_ptr0 + pid * 3 + 2)
    
    # Compute normalization factor
    eps = 1e-5
    rstd = 1 / tl.sqrt(var / count + eps)
    
    # Normalize and apply affine transformation
    for col in range(0, D, RBLOCK):
        cols = col + col_start
        mask = cols < D
        
        x = tl.load(primals_3_ptr + row_idx * stride_s3_0 + cols * stride_s3_1, mask=mask, other=0.0)
        gamma = tl.load(primals_1_ptr + cols * stride_s1_0, mask=mask, other=1.0)
        beta = tl.load(primals_2_ptr + cols * stride_s2_0, mask=mask, other=0.0)
        
        # Normalize
        y = (x - mean) * rstd
        # Apply scale and bias
        y = y * gamma + beta
        
        # Store result
        tl.store(out_ptr1 + row_idx * stride_out_0 + cols * stride_out_1, y, mask=mask)

def fused_native_layer_norm(primals_3, primals_1, primals_2):
    """
    Wrapper function for the Triton layer normalization kernel
    
    Args:
        primals_3: Input tensor of shape (S, D)
        primals_1: Scale tensor of shape (D,)
        primals_2: Bias tensor of shape (D,)
    
    Returns:
        Tuple of (normalized tensor, mean, variance)
    """
    S, D = primals_3.shape
    
    # Allocate output buffers
    device = primals_3.device
    buf0 = torch.empty((S, 3), dtype=torch.float32, device=device)  # for mean, var, count
    buf4 = torch.empty_like(primals_3)  # normalized output
    
    # Configure grid and block sizes
    RBLOCK = min(128, D)
    grid = (S,)
    
    # Launch kernel
    triton_red_fused_native_layer_norm_0[(grid,)](
        primals_3, primals_1, primals_2,
        buf0, buf4,
        primals_3.stride(0), primals_3.stride(1),
        primals_1.stride(0), primals_2.stride(0),
        buf4.stride(0), buf4.stride(1),
        S, D,
        RBLOCK=RBLOCK
    )
    
    # Extract statistics
    means = buf0[:, 0]
    variances = buf0[:, 1] / buf0[:, 2]  # normalize by count
    
    return buf4, means, variances
