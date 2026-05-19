import triton
import triton.language as tl

@triton.jit
def parallel_rebased_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, z_ptr,
    q_batch_stride, q_head_stride, q_seq_stride, q_feat_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_feat_stride,
    v_batch_stride, v_head_stride, v_seq_stride, v_feat_stride,
    o_batch_stride, o_head_stride, o_seq_stride, o_feat_stride,
    z_batch_stride, z_head_stride, z_seq_stride,
    BTL, BTS, BK, BV, scale, use_normalize,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(BTL, BLOCK_M)
    num_pid_n = tl.cdiv(BTS, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    b = group_id
    h = (pid % num_pid_in_group) // (num_pid_m * num_pid_n)
    pid_mn = pid % (num_pid_m * num_pid_n)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    q_ptrs = q_ptr + b * q_batch_stride + h * q_head_stride + tl.expand_dims(offs_m, 1) * q_seq_stride + tl.expand_dims(offs_k, 0) * q_feat_stride
    k_ptrs = k_ptr + b * k_batch_stride + h * k_head_stride + tl.expand_dims(offs_k, 1) * k_feat_stride + tl.expand_dims(offs_n, 0) * k_seq_stride
    v_ptrs = v_ptr + b * v_batch_stride + h * v_head_stride + tl.expand_dims(offs_n, 1) * v_seq_stride + tl.expand_dims(offs_k, 0) * v_feat_stride
    o_ptrs = o_ptr + b * o_batch_stride + h * o_head_stride + tl.expand_dims(offs_m, 1) * o_seq_stride + tl.expand_dims(offs_n, 0) * o_feat_stride
    z_ptrs = z_ptr + b * z_batch_stride + h * z_head_stride + tl.expand_dims(offs_m, 1) * z_seq_stride

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, BK, BLOCK_K):
        q = tl.load(q_ptrs)
        k = tl.load(k_ptrs)
        acc += tl.dot(q, k, allow_tf32=False)
        q_ptrs += BLOCK_K * q_feat_stride
        k_ptrs += BLOCK_K * k_feat_stride

    if use_normalize:
        z = tl.sum(acc, 1)
        z = tl.where(z != 0, z, 1.0)
        tl.store(z_ptrs, z)

    acc *= scale
    if use_normalize:
        acc /= z[:, None]

    tl.store(o_ptrs, acc.to(o_ptr.dtype))

@triton.jit
def _parallel_rebased_bwd_dq(
    q_ptr, k_ptr, v_ptr, do_ptr, dz_ptr, dq_ptr,
    q_batch_stride, q_head_stride, q_seq_stride, q_feat_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_feat_stride,
    v_batch_stride, v_head_stride, v_seq_stride, v_feat_stride,
    do_batch_stride, do_head_stride, do_seq_stride, do_feat_stride,
    dz_batch_stride, dz_head_stride, dz_seq_stride,
    dq_batch_stride, dq_head_stride, dq_seq_stride, dq_feat_stride,
    BTL, BTS, BK, BV, scale, use_normalize,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(BTL, BLOCK_M)
    num_pid_n = tl.cdiv(BTS, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    b = group_id
    h = (pid % num_pid_in_group) // (num_pid_m * num_pid_n)
    pid_mn = pid % (num_pid_m * num_pid_n)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    q_ptrs = q_ptr + b * q_batch_stride + h * q_head_stride + tl.expand_dims(offs_m, 1) * q_seq_stride + tl.expand_dims(offs_k, 0) * q_feat_stride
    k_ptrs = k_ptr + b * k_batch_stride + h * k_head_stride + tl.expand_dims(offs_k, 1) * k_feat_stride + tl.expand_dims(offs_n, 0) * k_seq_stride
    v_ptrs = v_ptr + b * v_batch_stride + h * v_head_stride + tl.expand_dims(offs_n, 1) * v_seq_stride + tl.expand_dims(offs_k, 0) * v_feat_stride
    do_ptrs = do_ptr + b * do_batch_stride + h * do_head_stride + tl.expand_dims(offs_m, 1) * do_seq_stride + tl.expand_dims(offs_n, 0) * do_feat_stride
    dz_ptrs = dz_ptr + b * dz_batch_stride + h * dz_head_stride + tl.expand_dims(offs_m, 1) * dz_seq_stride
    dq_ptrs = dq_ptr + b * dq_batch_stride + h * dq_head_stride + tl.expand_dims(offs_m, 1) * dq_seq_stride + tl.expand_dims(offs_k, 0) * dq_feat_stride

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n in range(0, BTS, BLOCK_N):
        k = tl.load(k_ptrs + n * k_seq_stride)
        do = tl.load(do_ptrs + n * do_seq_stride)
        dz = tl.load(dz_ptrs + n * dz_seq_stride)
        if use_normalize:
            do /= dz[:, None]
        acc += tl.dot(do, k, allow_tf32=False)
    acc *= scale
    tl.store(dq_ptrs, acc.to(dq_ptr.dtype))

@triton.jit
def _parallel_rebased_bwd_dkv(
    q_ptr, k_ptr, v_ptr, do_ptr, dz_ptr, dk_ptr, dv_ptr,
    q_batch_stride, q_head_stride, q_seq_stride, q_feat_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_feat_stride,
    v_batch_stride, v_head_stride, v_seq_stride, v_feat_stride,
    do_batch_stride, do_head_stride, do_seq_stride, do_feat_stride,
    dz_batch_stride, dz_head_stride, dz_seq_stride,
    dk_batch_stride, dk_head_stride, dk_seq_stride, dk_feat_stride,
    dv_batch_stride, dv_head_stride, dv_seq_stride, dv_feat_stride,
    BTL, BTS, BK, BV, scale, use_normalize,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(BTL, BLOCK_M)
    num_pid_n = tl.cdiv(BTS, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    b = group_id
    h = (pid % num_pid_in_group) // (num_pid_m * num_pid_n)
    pid_mn = pid % (num_pid_m * num_pid_n)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    q_ptrs = q_ptr + b * q_batch_stride + h * q_head_stride + tl.expand_dims(offs_m, 1) * q_seq_stride + tl.expand_dims(offs_k, 0) * q_feat_stride
    k_ptrs = k_ptr + b * k_batch_stride + h * k_head_stride + tl.expand_dims(offs_k, 1) * k_feat_stride + tl.expand_dims(offs_n, 0) * k_seq_stride
    v_ptrs = v_ptr + b * v_batch_stride + h * v_head_stride + tl.expand_dims(offs_n, 1) * v_seq_stride + tl.expand_dims(offs_k, 0
