import torch
import triton
import triton.language as tl

@triton.jit
def rms_matmul_rbe(
    x_ptr, w_ptr, rms_w_ptr, output_ptr,
    stride_x_batch, stride_x_m, stride_x_k,
    stride_w_n, stride_w_k,
    stride_output_batch, stride_output_m, stride_output_n,
    M, N, K,
    eps: tl.constexpr,
    THETA: tl.constexpr,
    use_rotary: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_batch = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_batch = pid_batch * stride_x_batch
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    sum_squares = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_ptr_k = x_ptr + offs_batch + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        x = tl.load(x_ptr_k, mask=x_mask, other=0.0)
        sum_squares += tl.sum(x * x, axis=1)
    
    rms = tl.math.rsqrt(sum_squares / K + eps)
    if rms_w_ptr is not None:
        rms_w = tl.load(rms_w_ptr + offs_m, mask=(offs_m < M), other=1.0)
        rms *= rms_w
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=x.dtype)
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_ptr_k = x_ptr + offs_batch + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        x = tl.load(x_ptr_k, mask=x_mask, other=0.0)
        x_norm = x * rms[:, None]
        
        w_ptr_k = w_ptr + offs_n[:, None] * stride_w_n + offs_k[None, :] * stride_w_k
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w = tl.load(w_ptr_k, mask=w_mask, other=0.0)
        
        acc += tl.dot(x_norm, tl.trans(w))
    
    if use_rotary:
        pos = offs_m[:, None]
        dim = tl.arange(0, BLOCK_SIZE_N // 2, dtype=tl.float32)
        inv_freq = 1.0 / (THETA ** (2 * dim / (BLOCK_SIZE_N)))
        freqs = pos * inv_freq
        cos = tl.cos(freqs)
        sin = tl.sin(freqs)
        
        acc = acc.to(tl.float32)
        acc_reshaped = tl.reshape(acc, (BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
        a = acc_reshaped * tl.reshape(cos, (BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 1))
        b = acc_reshaped * tl.reshape(sin, (BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 1))
        rotated = tl.stack([a[..., 0] - b[..., 1], a[..., 1] + b[..., 0]], axis=-1)
        rotated = tl.reshape(rotated, acc.shape)
        acc = rotated.to(acc.dtype)
    
    output_ptr += offs_batch + offs_m[:, None] * stride_output_m + offs_n[None, :] * stride_output_n
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptr, acc, mask=output_mask)

def rms_matmul_rbe_wrapper(x: torch.Tensor, w: torch.Tensor, rms_w: torch.Tensor = None, use_rotary: bool = False, theta: float = 10000.0, eps: float = 1e-6):
    assert x.dim() == 4, "Input x must be 4D: (batch, seq_len, num_heads, head_size)"
    B, M, H, K = x.shape
    N = w.shape[0]
    
    x_reshaped = x.reshape(B * M, H, K)
    output = torch.empty((B * M, H, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N']),
        B * H
    )
    
    rms_matmul_rbe[grid](
        x_reshaped, w, rms_w,
        output,
        x_reshaped.stride(0), x_reshaped.stride(1), x_reshaped.stride(2),
        w.stride(0), w.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        M, N, K,
        eps=eps,
        THETA=theta,
        use_rotary=use_rotary,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
    )
    
    output = output.reshape(B, M, H, N)
    return output
