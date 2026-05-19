import triton
import triton.language as tl
import torch

@triton.jit
def _attn_fwd_inner(
    acc, l_q, l_k, l_v,
    q_ptrs, k_ptrs, v_ptrs, o_ptrs,
    q_scale, k_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, STAGE: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(acc.shape[0], BLOCK_M)
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_m

    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load query and key scales
    q_scale = tl.load(q_scale + offs_m)
    k_scale = tl.load(k_scale + offs_n)
    
    # Initialize accumulators
    if STAGE == 0:
        l_q = tl.load(q_ptrs + offs_m[:, None] * BLOCK_DMODEL + offs_d[None, :])
        l_k = tl.load(k_ptrs + offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :])
        # Compute Q·K^T
        acc = tl.dot(l_q, tl.trans(l_k))
        # Apply scaling
        acc = acc * q_scale[:, None] * k_scale[None, :]
    
    elif STAGE == 1:
        # Load values
        l_v = tl.load(v_ptrs + offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :])
        # Apply softmax
        acc = tl.softmax(acc)
        # Compute attention output
        acc = tl.dot(acc, l_v)
        # Store output
        tl.store(o_ptrs + offs_m[:, None] * BLOCK_DMODEL + offs_d[None, :], acc)

@triton.jit
def _attn_fwd(
    q_ptrs, k_ptrs, v_ptrs, o_ptrs,
    q_scale, k_scale,
    M, N, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Shared memory buffers
    l_q = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_k = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    l_v = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Stage 0: Compute QK^T
    _attn_fwd_inner(acc, l_q, l_k, l_v, q_ptrs, k_ptrs, v_ptrs, o_ptrs,
                    q_scale, k_scale, BLOCK_M, BLOCK_N, BLOCK_DMODEL, 0)
    
    # Stage 1: Apply softmax and compute output
    _attn_fwd_inner(acc, l_q, l_k, l_v, q_ptrs, k_ptrs, v_ptrs, o_ptrs,
                    q_scale, k_scale, BLOCK_M, BLOCK_N, BLOCK_DMODEL, 1)

def forward(q, k, v, q_scale, k_scale):
    batch_size, seq_len, d_model = q.shape
    
    # Create output tensor
    o = torch.empty_like(q)
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(seq_len, META['BLOCK_M']) *
        triton.cdiv(seq_len, META['BLOCK_N']),
    )
    
    _attn_fwd[grid](
        q, k, v, o,
        q_scale, k_scale,
        seq_len, seq_len, d_model,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_DMODEL=64,
    )
    
    return o
