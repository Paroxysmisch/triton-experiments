import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, o_ptr,
    # Matrix dimensions
    batch_size, n_heads, seq_len, d_head,
    # Strides for the different matrices
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    # Scale factor
    sm_scale,
    # Block sizes (must be power of 2)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    # Optional parameters
    IS_CAUSAL: tl.constexpr,
    USE_FP8: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(seq_len, BLOCK_M)
    num_pid_n = tl.cdiv(seq_len, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, seq_len - first_pid_m * BLOCK_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize pointers to Q, K, V
    q_block_ptr = tl.make_block_ptr(
        q_ptr + group_id * stride_qz,
        (seq_len, d_head),
        (stride_qm, stride_qk),
        (offs_m, offs_d),
        (BLOCK_M, BLOCK_DMODEL),
        (1, 0)
    )
    k_block_ptr = tl.make_block_ptr(
        k_ptr + group_id * stride_kz,
        (d_head, seq_len),
        (stride_kk, stride_kn),
        (offs_d, offs_n),
        (BLOCK_DMODEL, BLOCK_N),
        (0, 1)
    )
    v_block_ptr = tl.make_block_ptr(
        v_ptr + group_id * stride_vz,
        (seq_len, d_head),
        (stride_vn, stride_vk),
        (offs_n, offs_d),
        (BLOCK_N, BLOCK_DMODEL),
        (1, 0)
    )

    # Load Q, K, V blocks
    q = tl.load(q_block_ptr)
    k = tl.load(k_block_ptr)
    v = tl.load(v_block_ptr)

    # Compute attention scores
    scores = tl.dot(q, k)
    scores = scores * sm_scale

    # Apply causal mask if needed
    if IS_CAUSAL:
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        scores = tl.where(causal_mask, scores, float("-inf"))

    # Apply softmax
    scores = tl.softmax(scores)

    # Optional FP8 conversion
    if USE_FP8:
        scores = scores.to(tl.float8e4m3fn)

    # Compute output
    o = tl.dot(scores, v)

    # Store output
    o_block_ptr = tl.make_block_ptr(
        o_ptr + group_id * stride_oz,
        (seq_len, d_head),
        (stride_om, stride_ok),
        (offs_m, offs_d),
        (BLOCK_M, BLOCK_DMODEL),
        (1, 0)
    )
    tl.store(o_block_ptr, o)

def triton_fa(q, k, v, sm_scale=None, is_causal=False):
    # Shape constraints
    batch_size, n_heads, seq_len, d_head = q.shape
    assert k.shape == (batch_size, n_heads, seq_len, d_head)
    assert v.shape == (batch_size, n_heads, seq_len, d_head)
    
    # Compute scale if not provided
    if sm_scale is None:
        sm_scale = 1.0 / (d_head ** 0.5)

    # Allocate output
    o = torch.empty_like(q)

    # Configure block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = min(128, triton.next_power_of_2(d_head))

    # Configure grid
    grid = (batch_size * n_heads * triton.cdiv(seq_len, BLOCK_M),)

    # Launch kernel
    _fwd_kernel[grid](
        q, k, v, o,
        batch_size, n_heads, seq_len, d_head,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=is_causal,
        USE_FP8=False,
        num_warps=4,
        num_stages=2
    )
    
    return o
