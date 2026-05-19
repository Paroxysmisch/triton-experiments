import triton
import triton.language as tl
import torch

@triton.jit
def ff_llama_kernel(
    # Pointers to matrices
    x_ptr, w1_ptr, w3_ptr, rms_w_ptr, output_ptr,
    # Matrix dimensions
    M, K, N,
    # Strides
    x_stride_m, x_stride_k,
    w1_stride_k, w1_stride_n,
    w3_stride_k, w3_stride_n,
    output_stride_m, output_stride_n,
    # Parameters
    EPS: tl.constexpr,
    USE_FP8: tl.constexpr,
    # Tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid = pid_m * tl.cdiv(N, BLOCK_SIZE_N) + pid_n

    # Offsets for the block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_row_ptrs = x_ptr + offs_m[:, None] * x_stride_m + offs_k[None, :] * x_stride_k
    x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

    # Compute sum of squares for RMS normalization
    sum_sq = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K + offs_k
        x = tl.load(x_row_ptrs, mask=x_mask & (k_offs[None, :] < K), other=0.0)
        sum_sq += tl.sum(x * x, axis=1)
        x_row_ptrs += BLOCK_SIZE_K * x_stride_k

    rms = tl.sqrt(sum_sq / K + EPS)

    # Reset pointers for the second pass
    x_row_ptrs = x_ptr + offs_m[:, None] * x_stride_m + offs_k[None, :] * x_stride_k
    w1_col_ptrs = w1_ptr + offs_k[:, None] * w1_stride_k + offs_n[None, :] * w1_stride_n
    w3_col_ptrs = w3_ptr + offs_k[:, None] * w3_stride_k + offs_n[None, :] * w3_stride_n

    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K + offs_k
        x = tl.load(x_row_ptrs, mask=x_mask & (k_offs[None, :] < K), other=0.0)
        rms_w = tl.load(rms_w_ptr + k_offs, mask=k_offs < K, other=1.0)
        
        # Apply RMS normalization
        x_norm = x * (rms_w[None, :] / rms[:, None])
        
        if USE_FP8:
            w1 = tl.load(w1_col_ptrs, mask=(k_offs[:, None] < K) & (offs_n[None, :] < N), other=0.0, eviction_policy="evict_first").to(tl.float16)
            w3 = tl.load(w3_col_ptrs, mask=(k_offs[:, None] < K) & (offs_n[None, :] < N), other=0.0, eviction_policy="evict_first").to(tl.float16)
        else:
            w1 = tl.load(w1_col_ptrs, mask=(k_offs[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            w3 = tl.load(w3_col_ptrs, mask=(k_offs[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        
        # Accumulate matrix products
        acc1 += tl.dot(x_norm, w1, allow_tf32=True)
        acc2 += tl.dot(x_norm, w3, allow_tf32=True)
        
        x_row_ptrs += BLOCK_SIZE_K * x_stride_k
        w1_col_ptrs += BLOCK_SIZE_K * w1_stride_k
        w3_col_ptrs += BLOCK_SIZE_K * w3_stride_k

    # Apply SiLU and element-wise multiplication
    silu_acc1 = acc1 * tl.sigmoid(acc1)
    output = silu_acc1 * acc2

    # Write output
    offs_output = offs_m[:, None] * output_stride_m + offs_n[None, :] * output_stride_n
    output_ptrs = output_ptr + offs_output
    tl.store(output_ptrs, output, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def kernel_ff(
    x: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    rms_w: torch.Tensor,
    EPS: float = 1e-5
) -> torch.Tensor:
    # Check dimensions
    assert x.dim() == 2, "x must be 2D"
    M, K = x.shape
    assert w1.dim() == 2 and w1.shape[0] == K, "w1 must be (K, N)"
    N = w1.shape[1]
    assert w3.shape == (K, N), "w3 must be (K, N)"
    assert rms_w.dim() == 1 and rms_w.shape[0] == K, "rms_w must be (K,)"
    
    # Check if using FP8
    USE_FP8 = w1.dtype in [torch.float8_e5m2, torch.float8_e4m3fn]
    
    # Ensure contiguous and transpose if needed
    x = x.contiguous()
    w1 = w1.contiguous()
    w3 = w3.contiguous()
    rms_w = rms_w.contiguous()
    
    # Output tensor
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Define grid lambda
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_SIZE_M']),
            triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
    
    # Launch kernel
    ff_llama_kernel[grid](
        x, w1, w3, rms_w, output,
        M, K, N,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w3.stride(0), w3.stride(1),
        output.stride(0), output.stride(1),
        EPS=EPS,
        USE_FP8=USE_FP8,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=64,
        num_warps=4,
        num_stages=3
    )
    
    return output
