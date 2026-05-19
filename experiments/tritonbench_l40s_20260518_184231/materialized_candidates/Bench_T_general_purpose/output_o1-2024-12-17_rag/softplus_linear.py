import torch
import triton
import triton.language as tl

@triton.jit
def _softplus_linear_kernel(
    A_ptr,          # [N, K]
    B_ptr,          # [M, K]
    Bias_ptr,       # [M] or None
    C_ptr,          # [N, M]
    N, M, K,
    stride_an, stride_ak,
    stride_bm, stride_bk,
    stride_cn, stride_cm,
    beta, threshold,
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program IDs for 2D launch grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create fragment IDs
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create pointer to the output block
    c_ptrs = C_ptr + offs_m[:, None] * stride_cn + offs_n[None, :] * stride_cm
    
    # Accumulator for partial sums
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K in steps of BLOCK_K
    # Each step computes a partial product
    for k_offs in range(0, K, BLOCK_K):
        # Offsets for K dimension in this iteration
        k_range = tl.arange(0, BLOCK_K)
        
        # A_ptr: [N, K], we index (offs_m, k_offs + k_range)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_an) + (k_offs + k_range[None, :]) * stride_ak
        
        # B_ptr: [M, K], we index (k_offs + k_range, offs_n)
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bm) + (k_offs + k_range[:, None]) * stride_bk
        
        # Load and convert
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < N) & (k_offs + k_range[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < M) & (k_offs + k_range[:, None] < K), other=0.0)
        
        # Transpose b so shapes match for matmul
        b = tl.trans(b)
        
        # Compute partial matmul
        acc += tl.dot(a, b)
    
    # Add bias if present
    if has_bias:
        # offs_n indexes the "M" dimension
        bias_ptrs = Bias_ptr + offs_n[None, :]
        bias_vals = tl.load(bias_ptrs, mask=offs_n[None, :] < M, other=0.0)
        acc += bias_vals
    
    # Apply Softplus with threshold
    # out = x if x > threshold else (1/beta) * log(1 + exp(beta*x))
    # We'll do it element-wise.
    # For numerical stability, if x > threshold, we approximate out ~ x.
    x = acc
    mask_over = x > threshold
    # Softplus for values <= threshold
    x_softplus = (1.0 / beta) * tl.log(1.0 + tl.exp(beta * x))
    out = tl.where(mask_over, x, x_softplus)
    
    # Store the result
    tl.store(c_ptrs, out, mask=(offs_m[:, None] < N) & (offs_n[None, :] < M))

def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    """
    Applies a linear transformation (input @ weight.T [+ bias]) and then
    an element-wise Softplus activation, with optional numerical stability
    for values above a specified threshold.
    
    softplus_linear(input, weight, bias=None, beta=1, threshold=20) -> Tensor
    """
    # Shapes and device
    assert input.dim() == 2, "Input must be 2D"
    assert weight.dim() == 2, "Weight must be 2D"
    N, in_features = input.shape
    out_features, w_in_features = weight.shape
    assert in_features == w_in_features, "Input and weight dimensions must match"
    device = input.device
    
    # Prepare output
    out = torch.empty((N, out_features), device=device, dtype=input.dtype)
    
    # Strides
    # A [N, K] => input
    # B [M, K] => weight (but we are passing weight as if shape [M, K], so let's
    # treat it consistently, or we can do weight = weight if stored that way?)
    # The user might store weight as [out_features, in_features], so let's define:
    A_stride_an = input.stride(0)
    A_stride_ak = input.stride(1)
    B_stride_bm = weight.stride(0)
    B_stride_bk = weight.stride(1)
    C_stride_cn = out.stride(0)
    C_stride_cm = out.stride(1)
    
    has_bias = (bias is not None)
    if has_bias:
        assert bias.shape[0] == out_features, "Bias must be of size [out_features]"
        bias = bias.to(input.dtype)
    
    # Define block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    # Launch kernel
    grid = (
        (N + BLOCK_M - 1) // BLOCK_M,
        (out_features + BLOCK_N - 1) // BLOCK_N
    )
    
    _softplus_linear_kernel[grid](
        input, weight, bias if has_bias else torch.empty(0, device=device, dtype=input.dtype),
        out,
        N, out_features, in_features,
        A_stride_an, A_stride_ak,
        B_stride_bm, B_stride_bk,
        C_stride_cn, C_stride_cm,
        beta, threshold,
        has_bias=has_bias,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return out
