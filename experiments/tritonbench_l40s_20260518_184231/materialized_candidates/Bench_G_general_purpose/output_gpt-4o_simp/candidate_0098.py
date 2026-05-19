import triton
import triton.language as tl

# Define compile-time constants
BLOCK_HEAD = 1
BLOCK_SEQ = 128
BLOCK_DMODEL = 64

@triton.jit
def _rotary_kernel(
    Q_ptr, K_ptr, Cos_ptr, Sin_ptr,
    stride_qh, stride_qs, stride_qd,
    stride_kh, stride_ks, stride_kd,
    stride_cos, stride_sin,
    max_total_len, HEAD_Q, HEAD_K,
    **meta
):
    # Define block indices
    head_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)

    # Compute offsets
    q_offset = head_idx * stride_qh + seq_idx * stride_qs
    k_offset = head_idx * stride_kh + seq_idx * stride_ks
    cos_offset = seq_idx * stride_cos
    sin_offset = seq_idx * stride_sin

    # Load data for Q and K
    Q = tl.load(Q_ptr + q_offset + tl.arange(0, BLOCK_DMODEL), mask=tl.arange(0, BLOCK_DMODEL) < max_total_len, other=0.0)
    K = tl.load(K_ptr + k_offset + tl.arange(0, BLOCK_DMODEL), mask=tl.arange(0, BLOCK_DMODEL) < max_total_len, other=0.0)

    # Load Cos and Sin
    Cos = tl.load(Cos_ptr + cos_offset + tl.arange(0, BLOCK_DMODEL), mask=tl.arange(0, BLOCK_DMODEL) < max_total_len, other=0.0)
    Sin = tl.load(Sin_ptr + sin_offset + tl.arange(0, BLOCK_DMODEL), mask=tl.arange(0, BLOCK_DMODEL) < max_total_len, other=0.0)

    # Perform rotary transformation
    Q_rotated = Q * Cos - K * Sin
    K_rotated = K * Cos + Q * Sin

    # Store the results back
    tl.store(Q_ptr + q_offset + tl.arange(0, BLOCK_DMODEL), Q_rotated, mask=tl.arange(0, BLOCK_DMODEL) < max_total_len)
    tl.store(K_ptr + k_offset + tl.arange(0, BLOCK_DMODEL), K_rotated, mask=tl.arange(0, BLOCK_DMODEL) < max_total_len)

def rotary_emb_fwd(Q, K, Cos, Sin, max_total_len, HEAD_Q, HEAD_K):
    # Get the shape of the input tensors
    num_heads, seq_len, d_model = Q.shape

    # Determine grid size
    grid = (HEAD_Q, (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ)

    # Launch kernel
    triton.run(
        _rotary_kernel,
        grid=grid,
        num_warps=4,  # Choose number of warps based on d_model
        args=[
            Q, K, Cos, Sin,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            Cos.stride(0), Sin.stride(0),
            max_total_len, HEAD_Q, HEAD_K
        ],
        meta={
            'BLOCK_HEAD': BLOCK_HEAD,
            'BLOCK_SEQ': BLOCK_SEQ,
            'BLOCK_DMODEL': BLOCK_DMODEL
        }
    )
