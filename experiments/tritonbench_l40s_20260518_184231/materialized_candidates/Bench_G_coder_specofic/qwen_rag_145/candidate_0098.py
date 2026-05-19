import triton.language as tl
from triton import jit, triton

BLOCK_HEAD = 16  # number of heads processed by each CUDA thread
BLOCK_SEQ = 8  # number of sequences processed by each CUDA thread
BLOCK_DMODEL = 8  # number of dmodel processed by each CUDA thread

@jit
def _rotary_kernel(
    Q, K, Cos, Sin,
    cos_offsets, sin_offsets,
    Q_stride_head, Q_stride_seq,
    K_stride_head, K_stride_seq,
    dmodel, seq_len, num_heads
):
    pid_head = tl.program_id(0)
    pid_seq = tl.program_id(1)

    q_offset = pid_head * Q_stride_head + pid_seq * Q_stride_seq
    k_offset = pid_head * K_stride_head + pid_seq * K_stride_seq

    cp = cos_offsets[pid_head] * dmodel // 2
    sp = sin_offsets[pid_head] * dmodel // 2
    
    # Load Q and K
    q0 = tl.load(Q + q_offset, mask=q_mask)
    q1 = tl.load(Q + q_offset + dmodel // 2, mask=q_mask)
    k0 = tl.load(K + k_offset, mask=k_mask)
    k1 = tl.load(K + k_offset + dmodel // 2, mask=k_mask)

    # Apply rotary transformation
    q0 = q0 * Cos[cp] - q1 * Sin[sp]
    q1 = q0 * Sin[sp + dmodel // 2] + q1 * Cos[cp + dmodel // 2]
    k0 = k0 * Cos[cp] - k1 * Sin[sp]
    k1 = k0 * Sin[sp + dmodel // 2] + k1 * Cos[cp + dmodel // 2]

    # Store Q and K
    tl.store(Q + q_offset, q0, mask=q_mask)
    tl.store(Q + q_offset + dmodel // 2, q1, mask=q_mask)
    tl.store(K + k_offset, k0, mask=k_mask)
    tl.store(K + k_offset + dmodel // 2, k1, mask=k_mask)

def rotary_emb_fwd(Q, K, Cos, Sin, cos_offsets, sin_offsets, num_heads, seq_len, dmodel):

    # Numbers of warps
    nw_head = triton.next_power_of_2(num_heads)
    nw_seq = triton.next_power_of_2(seq_len)
    grid = (triton.next_power_of_2(nw_head), triton.next_power_of_2(nw_seq))

    # Strides
    Q_stride_head = dmodel * seq_len
    Q_stride_seq = dmodel
    K_stride_head = dmodel * seq_len
    K_stride_seq = dmodel

    # Call the kernel
    _rotary_kernel[grid](
        Q, K, Cos, Sin,
        cos_offsets, sin_offsets,
        Q_stride_head, Q_stride_seq,
        K_stride_head, K_stride_seq,
        dmodel, seq_len, num_heads
    )
