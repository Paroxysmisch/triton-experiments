import torch
import triton
import triton.language as tl
import math

@triton.jit
def rms_matmul_rbe(
    x_ptr, w_ptr, rms_w_ptr, out_ptr,
    M, N, K,
    stride_xb, stride_xm, stride_xk,
    stride_wb, stride_wk, stride_wn,
    stride_ob, stride_om, stride_on,
    start_token_position: tl.int32,
    USE_FP8: tl.constexpr,
    RBE_EPILOGUE: tl.constexpr,
    THETA: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_b = 1  # Assuming batch size of 1 for simplicity
    
    # Compute batch/row/col indices for this program instance
    pid_b = pid // (num_pid_m * num_pid_n)
    pid_m = (pid % (num_pid_m * num_pid_n)) // num_pid_n
    pid_n = (pid % (num_pid_m * num_pid_n)) % num_pid_n

    # Compute memory offsets for this program instance
    offs_b = pid_b
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over k dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_idx = k * BLOCK_SIZE_K + offs_k
        
        # Load x block and compute RMS normalization
        x_block_ptr = x_ptr + offs_b * stride_xb + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk
        x = tl.load(x_block_ptr, mask=k_idx[None, :] < K, other=0.0)
        if USE_FP8:
            x = tl.cast(x, tl.float16)
        
        # RMS Normalization
        x_squared = x * x
        rms = tl.sqrt(tl.mean(x_squared, axis=1) + EPS)
        x = x / rms[:, None]
        
        # Load weights and RMS weights
        w_block_ptr = w_ptr + offs_b * stride_wb + k_idx[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w = tl.load(w_block_ptr, mask=k_idx[:, None] < K, other=0.0)
        if USE_FP8:
            w = tl.cast(w, tl.float16)
            
        rms_w = tl.load(rms_w_ptr + offs_n)
        
        # Compute matrix multiplication
        acc += tl.dot(x, w)
    
    # Apply RMS weight scaling
    acc = acc * rms_w[None, :]
    
    # Apply rotary embeddings if enabled
    if RBE_EPILOGUE:
        pos = start_token_position + offs_m
        freq = 1.0 / (THETA ** (tl.arange(0, N) / N))
        freq = tl.broadcast_to(freq, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        cos = tl.cos(pos[:, None] * freq)
        sin = tl.sin(pos[:, None] * freq)
        acc = acc * cos + tl.roll(acc, 1, axis=1) * sin
    
    # Store output
    out_ptr = out_ptr + offs_b * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptr, acc, mask=offs_m[:, None] < M and offs_n[None, :] < N)

@triton.jit
def rms_matmul_rbe_qkv(
    x_ptr, q_w_ptr, k_w_ptr, v_w_ptr,
    q_rms_w_ptr, k_rms_w_ptr, v_rms_w_ptr,
    q_out_ptr, k_out_ptr, v_out_ptr,
    M, N, K, start_token_position,
    **kwargs
):
    # Process Q matrix
    rms_matmul_rbe(
        x_ptr, q_w_ptr, q_rms_w_ptr, q_out_ptr,
        M, N, K, start_token_position,
        **kwargs
    )
    
    # Process K matrix
    rms_matmul_rbe(
        x_ptr, k_w_ptr, k_rms_w_ptr, k_out_ptr,
        M, N, K, start_token_position,
        **kwargs
    )
    
    # Process V matrix
    rms_matmul_rbe(
        x_ptr, v_w_ptr, v_rms_w_ptr, v_out_ptr,
        M, N, K, start_token_position,
        **kwargs
    )

def rms_matmul_rbe_qkv_wrapper(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    q_rms_weight: torch.Tensor,
    k_rms_weight: torch.Tensor,
    v_rms_weight: torch.Tensor,
    start_token_position: int = 0,
    use_fp8: bool = False,
    rbe_epilogue: bool = True,
    theta: float = 10000.0,
    eps: float = 1e-6,
    BLOCK_SIZE_M: int = 64,
    BLOCK_SIZE_N: int = 64,
    BLOCK_SIZE_K: int = 32
):
    # Input validation and shape checking
    assert x.dim() == 3, "Input tensor must be 3D (batch, seq_len, hidden_dim)"
    batch_size, seq_len, hidden_dim = x.shape
    
    # Ensure all weights have compatible shapes
    assert q_weight.shape == k_weight.shape == v_weight.shape
    assert q_weight.shape[0] == hidden_dim
    
    # Output tensors
    device = x.device
    dtype = torch.float16 if use_fp8 else x.dtype
    
    q = torch.empty((batch_size, seq_len, q_weight.shape[1]), device=device, dtype=dtype)
    k = torch.empty((batch_size, seq_len, k_weight.shape[1]), device=device, dtype=dtype)
    v = torch.empty((batch_size, seq_len, v_weight.shape[1]), device=device, dtype=dtype)
    
    # Compute grid
    grid = lambda META: (
        triton.cdiv(seq_len, META['BLOCK_SIZE_M']) *
        triton.cdiv(q_weight.shape[1], META['BLOCK_SIZE_N']) *
        batch_size,
    )
    
    # Launch kernel
    rms_matmul_rbe_qkv[grid](
        x, q_weight, k_weight, v_weight,
        q_rms_weight, k_rms_weight, v_rms_weight,
        q, k, v,
        seq_len, q_weight.shape[1], hidden_dim,
        x.stride(0), x.stride(1), x.stride(2),
        q_weight.stride(0), q_weight.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        start_token_position,
        USE_FP8=use_fp8,
        RBE_EPILOGUE=rbe_epilogue,
        THETA=theta,
        EPS=eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return q, k, v
