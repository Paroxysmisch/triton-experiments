import triton
import triton.language as tl

@triton.jit
def triton_red_fused_native_layer_norm_0(
    # Pointers to tensors
    primals_3_ptr, primals_1_ptr, primals_2_ptr,
    out_ptr1, out_ptr0,
    # Shape parameters
    S, D,
    # Strides for input tensor
    stride_primals_3_s, stride_primals_3_d,
    # Strides for scale and bias
    stride_primals_1_d, stride_primals_2_d,
    # Strides for output tensor
    stride_out1_s, stride_out1_d,
    # Optional parameters
    eps: tl.constexpr,
    # Tile sizes
    RBLOCK: tl.constexpr,  # Reduction block size for Welford
    BLOCK_SIZE: tl.constexpr,  # Block size for normalization
):
    # Row index
    pid = tl.program_id(0)
    
    # Initialize pointers for current row
    row_start = pid * stride_primals_3_s
    
    # Accumulators for Welford algorithm
    mean = 0.0
    m2 = 0.0
    weight = 0.0

    # Loop over the row to compute mean and variance
    for offset in range(0, D, RBLOCK):
        cols = offset + tl.arange(0, RBLOCK)
        mask = cols < D
        
        # Load data chunk
        x_ptr = primals_3_ptr + row_start + cols * stride_primals_3_d
        x = tl.load(x_ptr, mask=mask, other=0.0)
        
        # Update Welford statistics
        chunk_mean, chunk_m2, chunk_weight = tl.extra_ops.welford(x, mask)
        mean, m2, weight = tl.extra_ops.welford_combine(
            mean, m2, weight,
            chunk_mean, chunk_m2, chunk_weight
        )

    # Compute final variance and inverse std
    variance = m2 / weight + eps
    inv_std = 1.0 / tl.sqrt(variance)
    
    # Store statistics (mean, variance, weight)
    stats_offset = pid * 3
    tl.store(out_ptr0 + stats_offset, mean)
    tl.store(out_ptr0 + stats_offset + 1, variance)
    tl.store(out_ptr0 + stats_offset + 2, weight)

    # Normalize the row
    for offset in range(0, D, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        
        # Load input data
        x_ptr = primals_3_ptr + row_start + cols * stride_primals_3_d
        x = tl.load(x_ptr, mask=mask)
        
        # Load scale and bias
        scale_ptr = primals_1_ptr + cols * stride_primals_1_d
        scale = tl.load(scale_ptr, mask=mask)
        bias_ptr = primals_2_ptr + cols * stride_primals_2_d
        bias = tl.load(bias_ptr, mask=mask)
        
        # Compute normalized values
        centered = x - mean
        normalized = centered * inv_std
        scaled = normalized * scale + bias
        
        # Store results
        out_ptr = out_ptr1 + pid * stride_out1_s + cols * stride_out1_d
        tl.store(out_ptr, scaled, mask=mask)

def fused_native_layer_norm(primals_3, primals_1, primals_2, eps=1e-5):
    # Ensure input is contiguous
    primals_3 = primals_3.contiguous()
    S, D = primals_3.shape
    
    # Create output tensors
    buf4 = torch.empty_like(primals_3)
    buf0 = torch.empty((S, 3), device=primals_3.device, dtype=primals_3.dtype)
    
    # Handle optional scale/bias
    if primals_1 is None:
        primals_1 = torch.ones(D, device=primals_3.device, dtype=primals_3.dtype)
    if primals_2 is None:
        primals_2 = torch.zeros(D, device=primals_3.device, dtype=primals_3.dtype)
    
    # Configure kernel parameters
    RBLOCK = 1024  # Tuning these values is crucial for performance
    BLOCK_SIZE = 512
    
    # Launch kernel
    grid = (S,)
    triton_red_fused_native_layer_norm_0[grid](
        primals_3, primals_1, primals_2,
        buf4, buf0,
        S, D,
        primals_3.stride(0), primals_3.stride(1),
        primals_1.stride(0),
        primals_2.stride(0),
        buf4.stride(0), buf4.stride(1),
        eps,
        RBLOCK=RBLOCK,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return buf4, buf0
