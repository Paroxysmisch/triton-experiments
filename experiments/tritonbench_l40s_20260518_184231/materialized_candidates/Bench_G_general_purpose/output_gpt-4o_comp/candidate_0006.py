import triton
import triton.language as tl

# Define constants for block sizes
BLOCK_M = 128  # Block size for the sequence dimension
BLOCK_N = 128  # Block size for the sequence dimension
BLOCK_DMODEL = 64  # Block size for the model dimension

@triton.jit
def _fwd_kernel_aligned(Q, K, V, B0, sm_scale, OUT, stride_qz, stride_qh, stride_qm, stride_qd,
                        stride_kz, stride_kh, stride_kn, stride_kd,
                        stride_vz, stride_vh, stride_vn, stride_vd,
                        stride_b0z, stride_b0h, stride_b0m, stride_b0n,
                        stride_outz, stride_outh, stride_outm, stride_outd,
                        N_CTX, P_SEQ, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    # Compute block indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    m_idx = tl.program_id(2)

    # Compute pointers for Q, K, V, B0
    q_ptrs = Q + batch_idx * stride_qz + head_idx * stride_qh + m_idx * BLOCK_M * stride_qm
    k_ptrs = K + batch_idx * stride_kz + head_idx * stride_kh
    v_ptrs = V + batch_idx * stride_vz + head_idx * stride_vh
    b0_ptrs = B0 + batch_idx * stride_b0z + head_idx * stride_b0h + m_idx * BLOCK_M * stride_b0m

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=OUT.dtype.element_ty)

    # Iterate over sequence blocks
    for n_idx in range(0, N_CTX + P_SEQ, BLOCK_N):
        # Load Q, K, V, B0 blocks
        q = tl.load(q_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_qm + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qd)
        k = tl.load(k_ptrs + tl.arange(0, BLOCK_N)[:, None] * stride_kn + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_kd)
        v = tl.load(v_ptrs + tl.arange(0, BLOCK_N)[:, None] * stride_vn + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vd)
        b0 = tl.load(b0_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_b0m + tl.arange(0, BLOCK_N)[None, :] * stride_b0n)

        # Compute scaled dot-product
        qk = tl.dot(q, k, trans_b=True) * sm_scale
        qk += b0  # Add relative positional bias

        # Apply softmax
        qk = tl.softmax(qk, axis=1)

        # Accumulate weighted values
        acc += tl.dot(qk, v)

    # Store results
    out_ptrs = OUT + batch_idx * stride_outz + head_idx * stride_outh + m_idx * BLOCK_M * stride_outm
    tl.store(out_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_outm + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_outd, acc)

def _attention_rel_h_rel_w_kernel_aligned_device(Q, K, V, B0, sm_scale, OUT, N_CTX, P_SEQ, OUT_DTYPE):
    # Determine the number of blocks
    batch_size, num_heads, seq_len, d_model = Q.shape
    grid = (batch_size, num_heads, seq_len // BLOCK_M)

    # Launch the kernel
    _fwd_kernel_aligned[grid](
        Q, K, V, B0, sm_scale, OUT,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        B0.stride(0), B0.stride(1), B0.stride(2), B0.stride(3),
        OUT.stride(0), OUT.stride(1), OUT.stride(2), OUT.stride(3),
        N_CTX, P_SEQ, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL
    )
