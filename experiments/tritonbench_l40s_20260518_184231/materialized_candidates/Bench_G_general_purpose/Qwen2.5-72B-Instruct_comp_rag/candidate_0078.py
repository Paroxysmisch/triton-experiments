import triton
import triton.language as tl

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    q_ptr, k_ptr, g_ptr, A_ptr,
    q_batch_stride, q_head_stride, q_seq_stride, q_k_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_k_stride,
    g_batch_stride, g_head_stride, g_seq_stride, g_k_stride,
    A_batch_stride, A_head_stride, A_i_stride, A_j_stride,
    B, H, N_CTX, K, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_SIZE)
    num_pid_n = tl.cdiv(N_CTX, BLOCK_SIZE)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    first_pid_n = group_id * num_pid_n
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = first_pid_n + (pid % num_pid_n)

    offs_b = tl.arange(0, B)
    offs_h = tl.arange(0, H)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    q_offsets = (
        offs_b[:, None, None, None] * q_batch_stride +
        offs_h[None, :, None, None] * q_head_stride +
        offs_m[None, None, :, None] * q_seq_stride +
        tl.arange(0, K)[None, None, None, :] * q_k_stride
    )
    k_offsets = (
        offs_b[:, None, None, None] * k_batch_stride +
        offs_h[None, :, None, None] * k_head_stride +
        offs_n[None, None, :, None] * k_seq_stride +
        tl.arange(0, K)[None, None, None, :] * k_k_stride
    )
    g_offsets = (
        offs_b[:, None, None, None] * g_batch_stride +
        offs_h[None, :, None, None] * g_head_stride +
        offs_m[None, None, :, None] * g_seq_stride +
        tl.arange(0, K)[None, None, None, :] * g_k_stride
    )
    A_offsets = (
        offs_b[:, None, None, None] * A_batch_stride +
        offs_h[None, :, None, None] * A_head_stride +
        offs_m[None, None, :, None] * A_i_stride +
        offs_n[None, None, None, :] * A_j_stride
    )

    q = tl.load(q_ptr + q_offsets)
    k = tl.load(k_ptr + k_offsets)
    g = tl.load(g_ptr + g_offsets)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE):
        qk = tl.dot(q, k, allow_tf32=True)
        qk = tl.exp(qk * g)
        acc += qk

    tl.store(A_ptr + A_offsets, acc)

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra(
    q_ptr, k_ptr, g_ptr, A_ptr,
    q_batch_stride, q_head_stride, q_seq_stride, q_k_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_k_stride,
    g_batch_stride, g_head_stride, g_seq_stride, g_k_stride,
    A_batch_stride, A_head_stride, A_i_stride, A_j_stride,
    B, H, N_CTX, K, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_SIZE)
    num_pid_n = tl.cdiv(N_CTX, BLOCK_SIZE)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    first_pid_n = group_id * num_pid_n
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = first_pid_n + (pid % num_pid_n)

    offs_b = tl.arange(0, B)
    offs_h = tl.arange(0, H)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    q_offsets = (
        offs_b[:, None, None, None] * q_batch_stride +
        offs_h[None, :, None, None] * q_head_stride +
        offs_m[None, None, :, None] * q_seq_stride +
        tl.arange(0, K)[None, None, None, :] * q_k_stride
    )
    k_offsets = (
        offs_b[:, None, None, None] * k_batch_stride +
        offs_h[None, :, None, None] * k_head_stride +
        offs_n[None, None, :, None] * k_seq_stride +
        tl.arange(0, K)[None, None, None, :] * k_k_stride
    )
    g_offsets = (
        offs_b[:, None, None, None] * g_batch_stride +
        offs_h[None, :, None, None] * g_head_stride +
        offs_m[None, None, :, None] * g_seq_stride +
        tl.arange(0, K)[None, None, None, :] * g_k_stride
    )
    A_offsets = (
        offs_b[:, None, None, None] * A_batch_stride +
        offs_h[None, :, None, None] * A_head_stride +
        offs_m[None, None, :, None] * A_i_stride +
        offs_n[None, None, None, :] * A_j_stride
    )

    q = tl.load(q_ptr + q_offsets)
    k = tl.load(k_ptr + k_offsets)
    g = tl.load(g_ptr + g_offsets)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE):
        qk = tl.dot(q, k, allow_tf32=True)
        qk = tl.exp(qk * g)
        acc += qk

    tl.store(A_ptr + A_offsets, acc)
