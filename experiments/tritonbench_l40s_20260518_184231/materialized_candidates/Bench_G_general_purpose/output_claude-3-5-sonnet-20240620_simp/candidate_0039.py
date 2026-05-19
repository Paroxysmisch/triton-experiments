import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_int8kv(
    # Pointers to matrices
    Q, K, V, Out,
    # K/V scales for int8 quantization
    K_scale, V_scale,
    # Matrix dimensions
    batch, heads, seqlen_q, seqlen_k, dim,
    # Strides for memory access
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(seqlen_q, BLOCK_M)
    num_pid_n = tl.cdiv(seqlen_k, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, seqlen_q - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Load scales for current block
    k_scale = tl.load(K_scale + pid_n)
    v_scale = tl.load(V_scale + pid_n)
    
    # Pointers for Q, K, V blocks
    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * 1
    k_ptrs = K + offs_n[:, None] * stride_kn + offs_d[None, :] * 1
    v_ptrs = V + offs_n[:, None] * stride_vn + offs_d[None, :] * 1
    
    # Load and compute
    for d in range(0, dim, BLOCK_DMODEL):
        # Load Q block (fp16/fp32)
        q = tl.load(q_ptrs)
        # Load K block (int8) and dequantize
        k = tl.load(k_ptrs)
        k = (k.to(tl.float32) * k_scale)
        # Compute attention scores
        acc += tl.dot(q, k.transpose())
    
    # Scale and apply softmax
    acc = acc * (1.0 / tl.sqrt(dim))
    acc = tl.softmax(acc, axis=1)
    
    # Load V and compute output
    offs_d = tl.arange(0, BLOCK_DMODEL)
    for d in range(0, dim, BLOCK_DMODEL):
        # Load V block (int8) and dequantize
        v = tl.load(v_ptrs)
        v = (v.to(tl.float32) * v_scale)
        # Compute output
        o = tl.dot(acc, v)
        # Write output
        out_ptrs = Out + offs_m[:, None] * stride_om + (d + offs_d[None, :]) * 1
        tl.store(out_ptrs, o)

# Wrapper function
def context_attention_fwd_ppl_int8kv(q, k, v, k_scale, v_scale):
    """
    q: (batch, heads, seqlen_q, dim)
    k: (batch, heads, seqlen_k, dim)
    v: (batch, heads, seqlen_k, dim)
    k_scale, v_scale: (seqlen_k,)
    """
    batch, heads, seqlen_q, dim = q.shape
    seqlen_k = k.shape[2]
    
    # Allocate output
    out = torch.empty_like(q)
    
    # Configure meta-parameters
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_DMODEL = 32
    
    # Launch kernel
    grid = (triton.cdiv(seqlen_q, BLOCK_M) * triton.cdiv(seqlen_k, BLOCK_N) * batch * heads,)
    _fwd_kernel_int8kv[grid](
        q, k, v, out,
        k_scale, v_scale,
        batch, heads, seqlen_q, seqlen_k, dim,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
    )
    
    return out
