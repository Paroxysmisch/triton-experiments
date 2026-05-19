import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_int8kv(
    Q, K, V, Out,
    K_scale, V_scale, Softmax_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_pid_n = tl.cdiv(N_CTX, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Initialize offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize pointers
    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = K + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
    v_ptrs = V + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
    
    # Load Q, K, V
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX)
    k = tl.load(k_ptrs, mask=offs_n[None, :] < N_CTX)
    k = k * tl.load(K_scale + offs_n)  # Dequantize K
    v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX)
    v = v * tl.load(V_scale + offs_n)  # Dequantize V

    # Compute attention scores
    qk = tl.dot(q, k)
    qk = qk * Softmax_scale

    # Apply causal mask if needed
    if CAUSAL:
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))

    # Compute softmax
    qk_max = tl.max(qk, 1)
    qk = qk - qk_max[:, None]
    qk_exp = tl.exp(qk)
    qk_sum = tl.sum(qk_exp, 1)
    
    # Compute attention weights
    p = qk_exp / qk_sum[:, None]

    # Compute output
    out = tl.dot(p, v)
    
    # Store output
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    tl.store(out_ptrs, out, mask=offs_m[:, None] < N_CTX)

@triton.jit
def context_attention_fwd_ppl_int8kv(q, k, v, k_scale, v_scale, sm_scale, out,
                                    BLOCK_M: int = 128,
                                    BLOCK_N: int = 64,
                                    BLOCK_DMODEL: int = 64,
                                    num_warps: int = 4,
                                    num_stages: int = 3):
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    grid = (triton.cdiv(seq_len, BLOCK_M) * triton.cdiv(seq_len, BLOCK_N),)
    
    # Launch kernel
    _fwd_kernel_int8kv[grid](
        q, k, v, out,
        k_scale, v_scale, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch_size, num_heads, seq_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        CAUSAL=True,
        num_warps=num_warps,
        num_stages=num_stages
    )
    
    return out
