import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fwd_kernel_int8kv(
    Q, K, V, Out,
    K_scale, K_zp, V_scale, V_zp,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX, 
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr, 
    CAUSAL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize pointers
    q_ptrs = Q + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = K + off_hz * stride_kh + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
    v_ptrs = V + off_hz * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
    
    # Load Q
    q = tl.load(q_ptrs)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Load scaling factors for int8 KV
    k_scale_ptr = K_scale + off_hz * stride_kh
    k_zp_ptr = K_zp + off_hz * stride_kh
    v_scale_ptr = V_scale + off_hz * stride_vh
    v_zp_ptr = V_zp + off_hz * stride_vh
    
    k_scale = tl.load(k_scale_ptr)
    k_zp = tl.load(k_zp_ptr)
    v_scale = tl.load(v_scale_ptr)
    v_zp = tl.load(v_zp_ptr)
    
    # Compute attention scores
    for block_n in range(0, N_CTX, BLOCK_N):
        k_block_ptrs = k_ptrs + block_n * stride_kn
        v_block_ptrs = v_ptrs + block_n * stride_vn
        
        # Load and dequantize K, V
        k = (tl.load(k_block_ptrs).to(tl.float32) - k_zp) * k_scale
        v = (tl.load(v_block_ptrs).to(tl.float32) - v_zp) * v_scale
        
        # Compute QK
        qk = tl.dot(q, k)
        qk = qk * (1.0 / math.sqrt(BLOCK_DMODEL))
        
        # Apply causal mask if needed
        if IS_CAUSAL:
            mask = tl.arange(0, BLOCK_N) + block_n < offs_m[:, None]
            qk = tl.where(mask, qk, float("-inf"))
        
        # Compute softmax
        qk = tl.softmax(qk, axis=1)
        
        # Compute attention output
        acc += tl.dot(qk, v)
    
    # Store output
    out_ptrs = Out + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    tl.store(out_ptrs, acc)

def context_attention_fwd_ppl_int8kv(q, k, v, k_scale, k_zp, v_scale, v_zp, causal=False):
    batch_size, n_heads, seq_len, d_model = q.shape
    
    # Determine block sizes based on GPU capabilities
    BLOCK_M = min(128, seq_len)
    BLOCK_N = min(128, seq_len)
    BLOCK_DMODEL = d_model
    
    # Compute strides
    stride_qz, stride_qh, stride_qm, stride_qk = q.stride()
    stride_kz, stride_kh, stride_kn, stride_kk = k.stride()
    stride_vz, stride_vh, stride_vn, stride_vk = v.stride()
    
    # Initialize output tensor
    output = torch.empty_like(q)
    stride_oz, stride_oh, stride_om, stride_on = output.stride()
    
    # Configure grid
    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * n_heads)
    
    # Launch kernel
    _fwd_kernel_int8kv[grid](
        q, k, v, output,
        k_scale, k_zp, v_scale, v_zp,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_on,
        batch_size, n_heads, seq_len,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        CAUSAL=causal,
        IS_CAUSAL=causal,
    )
    
    return output
