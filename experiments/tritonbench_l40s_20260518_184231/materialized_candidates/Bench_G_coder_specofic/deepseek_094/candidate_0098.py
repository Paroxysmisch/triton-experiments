import triton as tl
import numpy as np

@triton.jit
def _rotary_kernel(Q_ptr, K_ptr, Cos_ptr, Sin_ptr, BLOCK_HEAD, BLOCK_SEQ, BLOCK_DMODEL):
    # Get current block indices
    head_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)

    # Calculate offsets
    q_offset = head_idx * BLOCK_SEQ * BLOCK_DMODEL + seq_idx * BLOCK_DMODEL
    k_offset = head_idx * BLOCK_SEQ * BLOCK_DMODEL + seq_idx * BLOCK_DMODEL

    # Load segments
    q0 = tl.load(Q_ptr + q_offset)
    q1 = tl.load(Q_ptr + q_offset + BLOCK_DMODEL)
    k0 = tl.load(K_ptr + k_offset)
    k1 = tl.load(K_ptr + k_offset + BLOCK_DMODEL)

    # Apply rotary transformation
    cos0 = tl.load(Cos_ptr + seq_idx * BLOCK_DMODEL)
    sin0 = tl.load(Cos_ptr + (seq_idx + 1) * BLOCK_DMODEL)
    cos1 = tl.load(Sin_ptr + seq_idx * BLOCK_DMODEL)
    sin1 = tl.load(Sin_ptr + (seq_idx + 1) * BLOCK_DMODEL)
    q0, q1 = q0 * cos0 - q1 * sin0, q0 * sin0 + q1 * cos0
    k0, k1 = k0 * cos1 - k1 * sin1, k0 * sin1 + k1 * cos1

    # Store transformed segments
    tl.store(Q_ptr + q_offset, q0)
    tl.store(Q_ptr + q_offset + BLOCK_DMODEL, q1)
    tl.store(K_ptr + k_offset, k0)
    tl.store(K_ptr + k_offset + BLOCK_DMODEL, k1)

def rotary_emb_fwd(Q, K, Cos, Sin):
    # Validate input shapes
    assert Q.shape == K.shape == Cos.shape == Sin.shape

    # Calculate execution grid dimensions
    num_heads = Q.shape[2]
    num_seq = Q.shape[1]
    BLOCK_DMODEL = Q.shape[3]
    grid = (num_heads, num_seq)

    # Determine number of warps
    num_warps = num_heads // 2 if num_heads % 2 == 0 else num_heads // 2 + 1

    # Invoke kernel
    _rotary_kernel[grid](Q.ctypes.data, K.ctypes.data, Cos.ctypes.data, Sin.ctypes.data, num_warps, num_seq, BLOCK_DMODEL)
