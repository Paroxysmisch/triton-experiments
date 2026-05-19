import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_int8kv(
    # Pointers to matrices
    Q, K, V, Out,
    # Pointers to scaling factors
    K_scale, V_scale,
    # Matrix dimensions
    batch_size, num_heads, seq_len, head_dim,
    # Strides for memory access
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    # Block sizes (constants)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(seq_len, BLOCK_M)
    num_pid_n = tl.cdiv(seq_len, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, num_pid_m - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize pointers to Q, K, V, and output
    q_ptrs = Q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + offs_n[None, :] * stride_ks + offs_d[:, None] * stride_kd
    v_ptrs = V + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    # Load Q block
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Load scaling factors for K and V
    k_scale_ptr = K_scale + pid_n
    v_scale_ptr = V_scale + pid_n
    k_scale = tl.load(k_scale_ptr)
    v_scale = tl.load(v_scale_ptr)
    
    # Load and dequantize K block
    k_int8 = tl.load(k_ptrs, mask=offs_n[None, :] < seq_len)
    k = k_int8 * k_scale
    
    # Compute attention scores
    scores = tl.dot(q, k)
    scores = scores * (1.0 / tl.sqrt(float(head_dim)))
    
    # Apply softmax
    scores = tl.softmax(scores)
    
    # Load and dequantize V block
    v_int8 = tl.load(v_ptrs, mask=offs_n[:, None] < seq_len)
    v = v_int8 * v_scale
    
    # Compute output
    acc += tl.dot(scores, v)
    
    # Store output
    out_ptrs = Out + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < seq_len)

def context_attention_fwd_ppl_int8kv(q, k, v, k_scale, v_scale):
    """
    Wrapper function for the int8 key-value attention kernel
    
    Args:
        q: Query tensor (batch_size, num_heads, seq_len, head_dim)
        k: Quantized key tensor (batch_size, num_heads, seq_len, head_dim)
        v: Quantized value tensor (batch_size, num_heads, seq_len, head_dim)
        k_scale: Key scaling factors
        v_scale: Value scaling factors
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Allocate output tensor
    output = torch.empty_like(q)
    
    # Configure block sizes
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_DMODEL = head_dim
    
    # Configure grid
    grid = (batch_size * num_heads * triton.cdiv(seq_len, BLOCK_M),)
    
    # Launch kernel
    _fwd_kernel_int8kv[grid](
        q, k, v, output,
        k_scale, v_scale,
        batch_size, num_heads, seq_len, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4
    )
    
    return output
