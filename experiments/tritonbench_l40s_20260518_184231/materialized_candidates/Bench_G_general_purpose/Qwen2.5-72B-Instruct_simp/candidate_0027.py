import triton
import triton.language as tl

# Constants
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
GROUP_SIZE_M = 8

@triton.jit
def _attn_fwd_inner(
    q_ptr, k_ptr, v_ptr, o_ptr,
    q_scale, k_scale,
    stride_qm, stride_kn, stride_vm, stride_vn, stride_om, stride_on,
    M, N, K,
    m_i, l_i,
    acc,
    offset_m, offset_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Pointers to the start of the block
    q_block_ptr = q_ptr + offset_m * stride_qm
    k_block_ptr = k_ptr + offset_n * stride_kn
    v_block_ptr = v_ptr + offset_n * stride_vm

    # Load the query and key blocks
    q = tl.load(q_block_ptr, mask=offset_m + tl.arange(0, BLOCK_M) < M, other=0.0)
    k = tl.load(k_block_ptr, mask=offset_n + tl.arange(0, BLOCK_N) < N, other=0.0)

    # Compute the attention scores
    q = q * q_scale
    k = k * k_scale
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        qk = tl.dot(q, k)
        qk = qk * (1.0 / tl.sqrt(K))
        qk = qk + m_i
        qk = tl.exp(qk - l_i)
        acc = acc + qk

    # Load the value block
    v = tl.load(v_block_ptr, mask=offset_n + tl.arange(0, BLOCK_N) < N, other=0.0)

    # Compute the weighted sum
    o = tl.dot(acc, v)

    # Store the result
    o_ptr = o_ptr + offset_m * stride_om + offset_n * stride_on
    tl.store(o_ptr, o, mask=offset_m + tl.arange(0, BLOCK_M) < M)

@triton.jit
def _attn_fwd(
    q_ptr, k_ptr, v_ptr, o_ptr,
    q_scale, k_scale,
    stride_qm, stride_kn, stride_vm, stride_vn, stride_om, stride_on,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N

    m_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        _attn_fwd_inner(
            q_ptr, k_ptr, v_ptr, o_ptr,
            q_scale, k_scale,
            stride_qm, stride_kn, stride_vm, stride_vn, stride_om, stride_on,
            M, N, K,
            m_i, l_i,
            acc,
            offset_m, offset_n,
            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M
        )

def forward(q, k, v, q_scale, k_scale):
    M, K = q.shape
    N, _ = v.shape
    o = triton.empty((M, N), dtype=q.dtype, device=q.device)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _attn_fwd[grid](
        q, k, v, o,
        q_scale, k_scale,
        q.stride(0), k.stride(0), v.stride(0), v.stride(1), o.stride(0), o.stride(1),
        M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M
    )

    return o
