import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    nheads, N_CTX,
    sm_scale: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    USE_FP8: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_pid_n = tl.cdiv(N_CTX, BLOCK_N)
    num_pid_in_block = num_pid_m * num_pid_n
    block_id = pid // num_pid_in_block
    pid_mn = pid % num_pid_in_block
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    offs_qm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + block_id * stride_qb + offs_qm[:, None] * stride_qm + offs_m[None, :] * stride_qh
    k_ptrs = K + block_id * stride_kb + offs_n[:, None] * stride_kn + offs_m[None, :] * stride_kh
    v_ptrs = V + block_id * stride_vb + offs_n[:, None] * stride_vn + offs_m[None, :] * stride_vh
    o_ptrs = Out + block_id * stride_ob + offs_qm[:, None] * stride_om + offs_n[None, :] * stride_oh

    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, BLOCK_DMODEL, 16):
        q_i = q[:, i:i+16]
        k_i = k[:, i:i+16]
        acc += tl.dot(q_i, k_i, trans_b=True)

    if IS_CAUSAL:
        mask = offs_qm[:, None] >= (offs_n[None, :] + 1)
        acc = tl.where(mask, acc, float('-inf'))

    acc *= sm_scale
    acc = tl.softmax(acc, axis=1)
    acc = acc.to(USE_FP8)

    o = tl.dot(acc, v)
    tl.store(o_ptrs, o)

import triton
import triton.language as tl

def triton_fa(Q, K, V, Out, sm_scale, IS_CAUSAL, BLOCK_M, BLOCK_N, BLOCK_DMODEL, USE_FP8):
    # Extract shapes and strides
    B, H, N_CTX, D = Q.shape
    stride_qb = Q.stride(0)
    stride_qh = Q.stride(1)
    stride_qm = Q.stride(2)
    stride_kb = K.stride(0)
    stride_kh = K.stride(1)
    stride_kn = K.stride(2)
    stride_vb = V.stride(0)
    stride_vh = V.stride(1)
    stride_vn = V.stride(2)
    stride_ob = Out.stride(0)
    stride_oh = Out.stride(1)
    stride_om = Out.stride(2)

    # Define grid and block sizes
    grid = (B * H * (N_CTX // BLOCK_M) * (N_CTX // BLOCK_N),)

    # Launch the kernel
    _fwd_kernel[grid](
        Q, K, V, Out,
        stride_qb, stride_qh, stride_qm,
        stride_kb, stride_kh, stride_kn,
        stride_vb, stride_vh, stride_vn,
        stride_ob, stride_oh, stride_om,
        H, N_CTX,
        sm_scale, IS_CAUSAL,
        BLOCK_M, BLOCK_N, BLOCK_DMODEL, USE_FP8
    )
