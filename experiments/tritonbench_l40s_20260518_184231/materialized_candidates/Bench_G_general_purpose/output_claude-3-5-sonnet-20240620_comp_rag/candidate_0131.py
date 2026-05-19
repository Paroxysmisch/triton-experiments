import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, o_ptr,
    # Matrix dimensions
    batch, heads, seq_len, dim,
    # Strides for the different matrices
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    # Scale for attention scores
    sm_scale,
    # Block sizes (passed as compile-time constants)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    # Optional params
    IS_CAUSAL: tl.constexpr,
    USE_FP8: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(seq_len, BLOCK_M)
    num_pid_n = tl.cdiv(seq_len, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Block pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize pointers to Q, K, V
    q_block_ptr = tl.make_block_ptr(
        q_ptr, (seq_len, dim),
        (stride_qs, 1),
        (pid_m * BLOCK_M, 0),
        (BLOCK_M, BLOCK_DMODEL),
        (1, 0)
    )
    k_block_ptr = tl.make_block_ptr(
        k_ptr, (dim, seq_len),
        (1, stride_ks),
        (0, pid_n * BLOCK_N),
        (BLOCK_DMODEL, BLOCK_N),
        (0, 1)
    )
    v_block_ptr = tl.make_block_ptr(
        v_ptr, (seq_len, dim),
        (stride_vs, 1),
        (pid_n * BLOCK_N, 0),
        (BLOCK_N, BLOCK_DMODEL),
        (1, 0)
    )

    # Load Q, K, V blocks
    q = tl.load(q_block_ptr)
    k = tl.load(k_block_ptr)
    v = tl.load(v_block_ptr)

    # Convert to FP8 if needed
    if USE_FP8:
        q = q.to(tl.float8e4m3fn)
        k = k.to(tl.float8e4m3fn)
        v = v.to(tl.float8e4m3fn)

    # Compute attention scores
    scores = tl.dot(q, k)
    scores = scores * sm_scale

    # Apply causal mask if needed
    if IS_CAUSAL:
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        scores = tl.where(causal_mask, scores, float("-inf"))

    # Apply softmax
    scores = tl.softmax(scores)

    # Compute output
    o = tl.dot(scores, v)

    # Store output
    o_block_ptr = tl.make_block_ptr(
        o_ptr, (seq_len, dim),
        (stride_os, 1),
        (pid_m * BLOCK_M, 0),
        (BLOCK_M, BLOCK_DMODEL),
        (1, 0)
    )
    tl.store(o_block_ptr, o)

def triton_fa(q, k, v, sm_scale=None, is_causal=False):
    """
    Wrapper function for the forward pass of FlashAttention using Triton
    """
    # Shape constraints and initialization
    batch, heads, seq_len, dim = q.shape
    assert k.shape == v.shape == (batch, heads, seq_len, dim)
    assert q.dtype == k.dtype == v.dtype
    
    # Set default scale if not provided
    if sm_scale is None:
        sm_scale = 1.0 / (dim ** 0.5)

    # Compute block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = min(128, triton.next_power_of_2(dim))

    # Allocate output
    o = torch.empty_like(q)

    # Grid configuration
    grid = (triton.cdiv(seq_len, BLOCK_M) * triton.cdiv(seq_len, BLOCK_N),)
    
    # Launch kernel
    _fwd_kernel[grid](
        q, k, v, o,
        batch, heads, seq_len, dim,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=is_causal,
        USE_FP8=q.dtype == torch.int8,
        num_warps=8 if BLOCK_DMODEL > 64 else 4,
        num_stages=2,
    )
    
    return o
