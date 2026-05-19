import torch
import triton
import triton.language as tl

@triton.jit
def rms_matmul_rbe(
    x_ptr, w_ptr, rms_w_ptr, out_ptr,
    M, N, K,
    stride_x_batch, stride_x_m, stride_x_k,
    stride_w_k, stride_w_n,
    stride_rms_w,
    stride_out_batch, stride_out_m, stride_out_n,
    start_token_position,
    USE_FP8: tl.constexpr,
    RBE_EPILOGUE: tl.constexpr,
    THETA: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID for batch and matrix blocks
    pid_batch = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)

    # Compute offsets
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Compute pointer offsets
    x_ptrs = x_ptr + (pid_batch * stride_x_batch + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n)
    
    # Initialize accumulators
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    x_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    # RMS normalization and matrix multiplication
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load and process input
        x = tl.load(x_ptrs)
        x_sum += tl.math.pow(x.to(tl.float32), 2)
        
        # Load and apply RMS weights
        rms_w = tl.load(rms_w_ptr + offs_k[None, :] * stride_rms_w)
        if USE_FP8:
            rms_w = rms_w.to(tl.float8e5, bitcast=True).to(tl.float16)
        x = x * rms_w
        
        # Load weights and perform matrix multiplication
        w = tl.load(w_ptrs)
        if USE_FP8:
            w = w.to(tl.float8e5, bitcast=True).to(tl.float32).to(tl.float16)
        accumulator += tl.dot(x, w)
        
        # Update pointers
        x_ptrs += BLOCK_SIZE_K * stride_x_k
        w_ptrs += BLOCK_SIZE_K * stride_w_k

    # Apply RMS normalization
    x_mean = tl.sum(x_sum, axis=1) / K + EPS
    x_norm = tl.math.rsqrt(x_mean)
    accumulator = accumulator * x_norm[:, None]

    # Store output
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (pid_batch * stride_out_batch + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    if RBE_EPILOGUE:
        # Apply rotary embeddings
        tl.store(out_ptrs, accumulator, mask=out_mask)
        tl.debug_barrier()
        rbe_triton(out_ptr, out_ptr, M, N, 
                   stride_out_batch, stride_out_m, stride_out_n,
                   stride_out_batch, stride_out_m, stride_out_n,
                   start_token_position, THETA, BLOCK_SIZE_M, BLOCK_SIZE_N)
    else:
        tl.store(out_ptrs, accumulator, mask=out_mask)

def rms_matmul_rbe_wrapper(x: torch.Tensor, weight: torch.Tensor, rms_w: torch.Tensor, 
                          use_rbe: bool, start_pos: int, n_heads: int, head_dim: int):
    # Input validation
    assert weight.dtype == rms_w.dtype
    assert weight.dtype in [torch.float16, torch.int8]
    
    # Shape setup
    batch, M, K = x.shape
    weight_t = weight.t()
    K_W, N = weight_t.shape
    assert K == K_W
    
    # Output tensor
    out = torch.empty((batch, M, N), dtype=weight_t.dtype, device=weight_t.device)
    out_ptr = triton.reinterpret(out, tl.float8e5 if out.dtype == torch.int8 else tl.float16)
    
    # Grid configuration
    grid = lambda META: (
        batch, 
        triton.cdiv(META["M"], META["BLOCK_SIZE_M"]) * triton.cdiv(META["N"], META["BLOCK_SIZE_N"])
    )
    
    # Launch kernel
    rms_matmul_rbe[grid](
        x_ptr=x,
        w_ptr=weight_t, 
        rms_w_ptr=rms_w, 
        out_ptr=out_ptr,
        M=M, N=N, K=K,
        stride_x_batch=x.stride(0), 
        stride_x_m=x.stride(1), 
        stride_x_k=x.stride(2),
        stride_w_k=weight_t.stride(0), 
        stride_w_n=weight_t.stride(1),
        stride_rms_w=rms_w.stride(0),
        stride_out_batch=out.stride(0), 
        stride_out_m=out.stride(1), 
        stride_out_n=out.stride(2),
        start_token_position=start_pos,
        USE_FP8=weight_t.dtype == torch.int8,
        RBE_EPILOGUE=use_rbe,
        THETA=10000.,
        EPS=1e-6,
        BLOCK_SIZE_M=16, 
        BLOCK_SIZE_N=64, 
        BLOCK_SIZE_K=64,
        num_stages=4, 
        num_warps=4
    )
    
    # Reshape output
    out = out.view(batch, M, n_heads, head_dim)
    return out
