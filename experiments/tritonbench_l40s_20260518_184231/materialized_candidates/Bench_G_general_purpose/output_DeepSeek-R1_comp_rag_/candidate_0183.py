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
    # Grid organization
    pid_batch = tl.program_id(0)
    pid = tl.program_id(1)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)

    # Block pointers and offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize pointers and accumulator
    x_ptrs = x_ptr + (pid_batch * stride_x_batch + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    x_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    # Main computation loop
    for _ in range(tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        rms_w = tl.load(rms_w_ptr + offs_k * stride_rms_w)
        
        # RMS normalization
        x_f32 = x.to(tl.float32)
        x_sum += tl.math.pow(x_f32, 2)
        x_norm = x_f32 * rms_w
        
        # FP8 conversion if needed
        if USE_FP8:
            x_norm = x_norm.to(tl.float8e5, bitcast=True).to(tl.float16)
            w = w.to(tl.float8e5, bitcast=True).to(tl.float32).to(tl.float16)
        
        accumulator += tl.dot(x_norm, w)
        x_ptrs += BLOCK_SIZE_K * stride_x_k
        w_ptrs += BLOCK_SIZE_K * stride_w_k

    # Final normalization and output
    rstd = tl.math.rsqrt(tl.sum(x_sum, axis=1) / K + EPS)
    accumulator *= rstd[:, None]
    
    # Store results
    out_ptrs = out_ptr + (pid_batch * stride_out_batch + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    tl.store(out_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    # Apply rotary embeddings if enabled
    if RBE_EPILOGUE:
        tl.debug_barrier()
        rbe_triton(out_ptr, out_ptr, M, N, 
                  stride_out_batch, stride_out_m, stride_out_n,
                  stride_out_batch, stride_out_m, stride_out_n,
                  start_token_position, THETA,
                  BLOCK_SIZE_M, BLOCK_SIZE_N)

def rms_matmul_rbe_wrapper(x: torch.Tensor, weight: torch.Tensor, rms_w: torch.Tensor, 
                          use_rbe: bool, start_pos: int, n_heads: int, head_dim: int):
    # Dimension checks and setup
    batch, M, K = x.shape
    weight_t = weight.t().contiguous()
    out = torch.empty((batch, M, weight_t.shape[1]), dtype=weight.dtype, device=x.device)

    # Kernel configuration
    grid = (batch, triton.cdiv(M, 16) * triton.cdiv(weight_t.shape[1], 64))
    
    # Kernel launch
    rms_matmul_rbe[grid](
        x, weight_t, rms_w, out,
        M, weight_t.shape[1], K,
        *x.stride(), *weight_t.stride(), rms_w.stride(0), *out.stride(),
        start_token_position=start_pos,
        USE_FP8=weight.dtype == torch.int8,
        RBE_EPILOGUE=use_rbe,
        THETA=10000.0,
        EPS=1e-6,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
        num_stages=4, num_warps=4
    )
    return out.view(batch, M, n_heads, head_dim)

@triton.jit
def rms_matmul_rbe_qkv(
    x_ptr,
    q_weight_ptr, k_weight_ptr, v_weight_ptr,
    rms_w_ptr,
    q_ptr, k_ptr, v_ptr,
    M, N, K,
    # Stride parameters...
    start_token_position,
    USE_FP8: tl.constexpr,
    THETA: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Call QKV variants with different parameters
    for weight_ptr, out_ptr, rbe_enabled in [
        (q_weight_ptr, q_ptr, True),
        (k_weight_ptr, k_ptr, True),
        (v_weight_ptr, v_ptr, False)
    ]:
        rms_matmul_rbe(
            x_ptr, weight_ptr, rms_w_ptr, out_ptr,
            M, N, K,
            # Pass stride parameters...
            start_token_position,
            USE_FP8,
            rbe_enabled,
            THETA,
            EPS,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
        )

def rms_matmul_rbe_qkv_wrapper(
    x: torch.Tensor,
    start_pos: int,
    q_weight: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor,
    rms_w: torch.Tensor,
    n_heads: int, head_dim: int
):
    # Tensor setup and validation
    batch, M, K = x.shape
    q, k, v = [torch.empty_like(x) for _ in range(3)]

    # Kernel configuration
    grid = (batch, triton.cdiv(M, 16) * triton.cdiv(q_weight.shape[1], 64))
    
    # Launch QKV kernel
    rms_matmul_rbe_qkv[grid](
        x,
        q_weight.t().contiguous(), k_weight.t().contiguous(), v_weight.t().contiguous(),
        rms_w,
        q, k, v,
        M, q_weight.shape[1], K,
        # Pass stride parameters...
        start_token_position=start_pos,
        USE_FP8=q_weight.dtype == torch.int8,
        THETA=10000.0,
        EPS=1e-6,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
        num_stages=4, num_warps=4
    )
    
    # Reshape outputs for attention heads
    return [t.view(batch, M, n_heads, head_dim) for t in (q, k, v)]
