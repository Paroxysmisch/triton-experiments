import triton
import triton.language as tl

@triton.jit
def _rotary_kernel(Q_ptr, K_ptr, Cos_ptr, Sin_ptr, BLOCK_HEAD, BLOCK_SEQ, BLOCK_DMODEL, stride_qh, stride_qs, stride_qd, stride_kh, stride_ks, stride_kd, n_heads, n_seqs, n_dmodel):
    # Program IDs
    pid_head = tl.program_id(0)
    pid_seq = tl.program_id(1)

    # Compute the offsets
    head_offset = pid_head * BLOCK_HEAD
    seq_offset = pid_seq * BLOCK_SEQ

    # Boundary checks
    head_mask = head_offset + tl.arange(0, BLOCK_HEAD) < n_heads
    seq_mask = seq_offset + tl.arange(0, BLOCK_SEQ) < n_seqs

    # Load Q and K segments
    q_offset = head_offset * stride_qh + seq_offset * stride_qs + tl.arange(0, BLOCK_DMODEL)
    k_offset = head_offset * stride_kh + seq_offset * stride_ks + tl.arange(0, BLOCK_DMODEL)

    q0 = tl.load(Q_ptr + q_offset, mask=head_mask[:, None] & seq_mask[None, :], other=0.0)
    k0 = tl.load(K_ptr + k_offset, mask=head_mask[:, None] & seq_mask[None, :], other=0.0)

    # Load Cos and Sin segments
    cos_offset = seq_offset * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)
    sin_offset = seq_offset * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)

    cos0 = tl.load(Cos_ptr + cos_offset, mask=seq_mask[None, :], other=1.0)
    sin0 = tl.load(Sin_ptr + sin_offset, mask=seq_mask[None, :], other=0.0)

    # Apply rotary transformation
    q1 = q0[:, 1::2]
    q0 = q0[:, ::2]

    out0 = q0 * cos0 - q1 * sin0
    out1 = q0 * sin0 + q1 * cos0

    # Store back the results
    tl.store(Q_ptr + q_offset, out0, mask=head_mask[:, None] & seq_mask[None, :])
    tl.store(K_ptr + k_offset, out1, mask=head_mask[:, None] & seq_mask[None, :])

def rotary_emb_fwd(Q, K, Cos, Sin, BLOCK_HEAD=1, BLOCK_SEQ=128, BLOCK_DMODEL=64):
    # Validate input shapes
    assert Q.shape == K.shape, "Q and K must have the same shape"
    assert Cos.shape == Sin.shape, "Cos and Sin must have the same shape"

    n_heads, n_seqs, n_dmodel = Q.shape

    # Determine grid size
    grid = (triton.cdiv(n_heads, BLOCK_HEAD), triton.cdiv(n_seqs, BLOCK_SEQ))

    # Launch the kernel
    _rotary_kernel[grid](
        Q, K, Cos, Sin,
        BLOCK_HEAD, BLOCK_SEQ, BLOCK_DMODEL,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        n_heads, n_seqs, n_dmodel,
        num_warps=4
    )
