import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,                     # Pointers to matrices
    stride_qm, stride_qh, stride_qk,  # Strides for Q matrix
    stride_km, stride_kh, stride_kk,  # Strides for K matrix
    stride_vm, stride_vh, stride_vk,  # Strides for V matrix
    stride_om, stride_oh, stride_ok,  # Strides for output matrix
    size_m, size_n, size_k,          # Matrix dimensions
    sm_scale,                         # Attention scale factor
    IS_CAUSAL: tl.constexpr,         # Enable causal attention
    USE_FP8: tl.constexpr,           # Enable FP8 computation
    BLOCK_M: tl.constexpr,           # Block size for M dimension
    BLOCK_N: tl.constexpr,           # Block size for N dimension
    BLOCK_DMODEL: tl.constexpr,      # Block size for K dimension
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(size_m, BLOCK_M)
    num_pid_n = tl.cdiv(size_n, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Initialize offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize pointers
    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = K + offs_n[None, :] * stride_km + offs_k[:, None] * stride_kk
    
    # Load Q and K blocks
    if USE_FP8:
        # Convert int8 to fp32 if using FP8
        q = tl.load(q_ptrs, mask=offs_m[:, None] < size_m, other=0).to(tl.float32)
        k = tl.load(k_ptrs, mask=offs_n[None, :] < size_n, other=0).to(tl.float32)
        if USE_FP8:
            q = q * (1.0 / 127.0)
            k = k * (1.0 / 127.0)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < size_m, other=0)
        k = tl.load(k_ptrs, mask=offs_n[None, :] < size_n, other=0)

    # Compute attention scores
    scores = tl.dot(q, k) * sm_scale

    # Apply causal mask if needed
    if IS_CAUSAL:
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        scores = tl.where(causal_mask, scores, float("-inf"))

    # Apply softmax
    scores = tl.softmax(scores, axis=1)

    # Load and multiply with values
    v_ptrs = V + offs_n[:, None] * stride_vm + offs_k[None, :] * stride_vk
    if USE_FP8:
        v = tl.load(v_ptrs, mask=offs_n[:, None] < size_n, other=0).to(tl.float32)
        if USE_FP8:
            v = v * (1.0 / 127.0)
    else:
        v = tl.load(v_ptrs, mask=offs_n[:, None] < size_n, other=0)

    # Compute output
    out = tl.dot(scores, v)

    # Store output
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, out, mask=offs_m[:, None] < size_m)

def triton_fa(q, k, v, sm_scale, is_causal=False):
    """
    Wrapper function for flash attention implementation using Triton
    
    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim)
        k: Key tensor of shape (batch, heads, seq_len, head_dim)
        v: Value tensor of shape (batch, heads, seq_len, head_dim)
        sm_scale: Scale factor for attention scores
        is_causal: Whether to apply causal masking
    """
    batch, heads, m_size, k_size = q.shape
    _, _, n_size, _ = k.shape
    
    # Handle data types
    use_fp8 = q.dtype == torch.int8
    
    # Set block sizes based on GPU architecture
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = k_size
    
    # Compute grid size
    grid = (triton.cdiv(m_size, BLOCK_M) * triton.cdiv(n_size, BLOCK_N),)
    
    # Determine number of warps based on block size
    num_warps = 8 if k_size >= 64 else 4
    
    # Reshape inputs for kernel
    q = q.reshape(batch * heads, m_size, k_size)
    k = k.reshape(batch * heads, n_size, k_size)
    v = v.reshape(batch * heads, n_size, k_size)
    
    # Allocate output
    output = torch.empty_like(q)
    
    # Launch kernel
    _fwd_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        m_size, n_size, k_size,
        sm_scale,
        IS_CAUSAL=is_causal,
        USE_FP8=use_fp8,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=2
    )
    
    # Reshape output back to original dimensions
    return output.reshape(batch, heads, m_size, k_size)
