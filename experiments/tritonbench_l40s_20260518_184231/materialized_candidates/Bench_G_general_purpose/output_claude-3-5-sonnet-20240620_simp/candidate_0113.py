import triton
import triton.language as tl
import torch

@triton.jit
def triton_red_fused_native_layer_norm_0(
    input_ptr, weight_ptr, bias_ptr,  # Pointers to input tensors
    mean_ptr, var_ptr, output_ptr,    # Pointers to output buffers
    stride_s, stride_d,               # Strides for accessing tensors
    N, D,                            # Dimensions: N=sequence length, D=hidden size
    eps: tl.float32,                 # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr         # Block size for parallelization
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute row index
    row_idx = pid
    
    # Initialize accumulators for mean and variance
    mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    m2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load column indices
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D
    
    # Compute base pointer for this row
    row_ptr = input_ptr + row_idx * stride_s
    
    # First pass: Compute mean
    x = tl.load(row_ptr + cols * stride_d, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / D
    
    # Second pass: Compute variance
    x = tl.load(row_ptr + cols * stride_d, mask=mask, other=0.0)
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / D
    
    # Store mean and variance
    tl.store(mean_ptr + row_idx, mean)
    tl.store(var_ptr + row_idx, var)
    
    # Normalize and apply weight/bias
    inv_std = 1 / tl.sqrt(var + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    
    # Load input again for normalization
    x = tl.load(row_ptr + cols * stride_d, mask=mask, other=0.0)
    out = (x - mean) * inv_std
    out = out * weight + bias
    
    # Store output
    out_ptr = output_ptr + row_idx * stride_s
    tl.store(out_ptr + cols * stride_d, out, mask=mask)

# Wrapper function
def fused_native_layer_norm(primals_1, primals_2, primals_3, eps=1e-5):
    # primals_1: weight
    # primals_2: bias
    # primals_3: input tensor
    
    # Get input dimensions
    S, D = primals_3.shape
    
    # Create output buffers
    buf0 = torch.empty((S,), device=primals_3.device, dtype=primals_3.dtype)  # mean
    buf3 = torch.empty((S,), device=primals_3.device, dtype=primals_3.dtype)  # variance
    buf4 = torch.empty_like(primals_3)  # normalized output
    
    # Compute strides
    stride_s = primals_3.stride(0)
    stride_d = primals_3.stride(1) if len(primals_3.stride()) > 1 else 1
    
    # Configure grid and block sizes
    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = (S,)
    
    # Launch kernel
    triton_red_fused_native_layer_norm_0[grid](
        primals_3, primals_1, primals_2,
        buf0, buf3, buf4,
        stride_s, stride_d,
        S, D,
        eps,
        BLOCK_SIZE,
    )
    
    return buf4, primals_3, buf0, buf3
