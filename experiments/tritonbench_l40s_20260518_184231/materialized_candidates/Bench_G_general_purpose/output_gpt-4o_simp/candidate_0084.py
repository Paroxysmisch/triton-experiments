import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def _fwd_kernel_aligned(Q_ptr, K_ptr, V_ptr, B0_ptr, Out_ptr,
                        stride_qz, stride_qh, stride_qm, stride_qd,
                        stride_kz, stride_kh, stride_kn, stride_kd,
                        stride_vz, stride_vh, stride_vn, stride_vd,
                        stride_b0z, stride_b0h, stride_b0m, stride_b0n,
                        stride_oz, stride_oh, stride_om, stride_od,
                        scale, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    
    pid_m = tl.program_id(0)
    pid_batch = tl.program_id(1)

    batch_z = pid_batch // stride_qh
    batch_h = pid_batch % stride_qh

    q_offset = batch_z * stride_qz + batch_h * stride_qh + pid_m * BLOCK_M * stride_qm
    k_offset = batch_z * stride_kz + batch_h * stride_kh
    v_offset = batch_z * stride_vz + batch_h * stride_vh
    b0_offset = batch_z * stride_b0z + batch_h * stride_b0h + pid_m * BLOCK_M * stride_b0m
    out_offset = batch_z * stride_oz + batch_h * stride_oh + pid_m * BLOCK_M * stride_om

    Q = tl.load(Q_ptr + q_offset + tl.arange(0, BLOCK_M)[:, None] * stride_qm + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qd)
    K = tl.load(K_ptr + k_offset + tl.arange(0, BLOCK_N)[:, None] * stride_kn + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_kd)
    V = tl.load(V_ptr + v_offset + tl.arange(0, BLOCK_N)[:, None] * stride_vn + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vd)
    B0 = tl.load(B0_ptr + b0_offset + tl.arange(0, BLOCK_M)[:, None] * stride_b0m + tl.arange(0, BLOCK_N)[None, :] * stride_b0n)

    # Compute QK^T
    QK = tl.dot(Q, K, trans_b=True) * scale
    QK += B0

    # Softmax
    QK_max = tl.max(QK, axis=1)
    QK_exp = tl.math.exp2(QK - QK_max[:, None])
    QK_sum = tl.sum(QK_exp, axis=1)
    QK_softmax = QK_exp / QK_sum[:, None]

    # Compute output
    Out = tl.dot(QK_softmax, V)
    tl.store(Out_ptr + out_offset + tl.arange(0, BLOCK_M)[:, None] * stride_om + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_od, Out)


# Define the wrapper function
def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, b0, out, scale):
    BLOCK_M = 128  # Example block size
    BLOCK_N = 128
    BLOCK_DMODEL = 64

    assert q.shape[2] % BLOCK_M == 0, "Q's third dimension must be divisible by BLOCK_M"
    assert q.shape[3] == BLOCK_DMODEL, "Q's fourth dimension must match BLOCK_DMODEL"
    assert k.shape[3] == BLOCK_DMODEL, "K's fourth dimension must match BLOCK_DMODEL"
    assert v.shape[3] == BLOCK_DMODEL, "V's fourth dimension must match BLOCK_DMODEL"
    
    grid = (q.shape[2] // BLOCK_M, q.shape[0] * q.shape[1])

    triton.kernel(_fwd_kernel_aligned, grid=grid)(
        q, k, v, b0, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        b0.stride(0), b0.stride(1), b0.stride(2), b0.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale, BLOCK_M, BLOCK_N, BLOCK_DMODEL
    )
