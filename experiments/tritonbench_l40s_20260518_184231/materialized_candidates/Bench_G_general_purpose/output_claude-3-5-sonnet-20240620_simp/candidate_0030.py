import triton
import triton.language as tl
import torch

@triton.jit
def triton_red_fused_native_layer_norm_no_welford_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr, mean_ptr, invstd_ptr,
    stride_batch, stride_dim,
    n_cols, eps,
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute batch index
    batch_idx = pid
    
    # Compute input/output pointers for this batch
    input_offset = batch_idx * stride_batch
    x_ptr = input_ptr + input_offset
    
    # Load input values for this block
    cols = tl.arange(0, XBLOCK)
    mask = cols < n_cols
    
    # Initialize accumulators for mean and variance
    mean = tl.zeros([XBLOCK], dtype=tl.float32)
    m2 = tl.zeros([XBLOCK], dtype=tl.float32)
    
    # First pass: compute mean
    row_idx = tl.arange(0, RBLOCK)
    for i in range(0, n_cols, RBLOCK):
        block_mask = row_idx[:, None] + i < n_cols
        x = tl.load(x_ptr + (row_idx[:, None] + i) * stride_dim, mask=block_mask, other=0.0)
        mean += tl.sum(x, axis=0)
    
    mean = mean / n_cols
    
    # Second pass: compute variance
    for i in range(0, n_cols, RBLOCK):
        block_mask = row_idx[:, None] + i < n_cols
        x = tl.load(x_ptr + (row_idx[:, None] + i) * stride_dim, mask=block_mask, other=0.0)
        diff = x - mean
        m2 += tl.sum(diff * diff, axis=0)
    
    # Compute inverse standard deviation
    var = m2 / n_cols
    invstd = 1 / tl.sqrt(var + eps)
    
    # Store mean and invstd
    tl.store(mean_ptr + batch_idx * n_cols + cols, mean, mask=mask)
    tl.store(invstd_ptr + batch_idx * n_cols + cols, invstd, mask=mask)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    
    # Third pass: normalize and apply weight/bias
    for i in range(0, n_cols, RBLOCK):
        block_mask = row_idx[:, None] + i < n_cols
        x = tl.load(x_ptr + (row_idx[:, None] + i) * stride_dim, mask=block_mask, other=0.0)
        x_normalized = (x - mean) * invstd
        x_normalized = x_normalized * weight + bias
        tl.store(output_ptr + batch_idx * stride_batch + (row_idx[:, None] + i) * stride_dim,
                x_normalized, mask=block_mask)

def fused_native_layer_norm_no_welford(primals_1, primals_2, primals_3, eps=1e-5):
    """
    Wrapper function for layer normalization using Triton.
    
    Args:
        primals_1: Input tensor to normalize
        primals_2: Weight tensor
        primals_3: Bias tensor
        eps: Small constant for numerical stability
    
    Returns:
        Tuple of (normalized tensor, mean tensor, inverse standard deviation tensor)
    """
    batch_size, seq_len, hidden_size = primals_1.shape
    
    # Allocate output tensors
    output = torch.empty_like(primals_1)
    mean = torch.empty((batch_size, hidden_size), device=primals_1.device)
    invstd = torch.empty((batch_size, hidden_size), device=primals_1.device)
    
    # Define block sizes
    XBLOCK = 32
    RBLOCK = 32
    
    # Launch kernel
    grid = (batch_size,)
    triton_red_fused_native_layer_norm_no_welford_kernel[grid](
        primals_1.contiguous(), primals_2.contiguous(), primals_3.contiguous(),
        output, mean, invstd,
        primals_1.stride(0), primals_1.stride(1),
        hidden_size, eps,
        XBLOCK=XBLOCK, RBLOCK=RBLOCK,
    )
    
    return output, mean, invstd
