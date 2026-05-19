import triton
import triton.language as tl

@triton.jit
def _rotary_kernel(
    Q_ptr, K_ptr, Cos_ptr, Sin_ptr,
    nheads, nseq, dmodel,
    stride_qH, stride_qS, stride_qD,
    stride_kH, stride_kS, stride_kD,
    stride_cosH, stride_cosS, stride_cosD,
    stride_sinH, stride_sinS, stride_sinD,
    BLOCK_HEAD: tl.constexpr, BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Program IDs for heads and sequence
    head_block = tl.program_id(0)
    seq_block = tl.program_id(1)

    # Offsets for heads, sequence, and dmodel
    offs_head = head_block * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    offs_seq = seq_block * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Expand dimensions for broadcasting
    heads = offs_head[:, None, None]  # shape: [BLOCK_HEAD, 1, 1]
    seqs = offs_seq[None, :, None]    # shape: [1, BLOCK_SEQ, 1]
    ds = offs_d[None, None, :]        # shape: [1, 1, BLOCK_DMODEL]

    # Mask for valid memory accesses
    head_mask = heads < nheads
    seq_mask = seqs < nseq
    d_mask = ds < dmodel
    mask = head_mask & seq_mask & d_mask

    # Compute pointer offsets
    q_offset = (heads * stride_qH) + (seqs * stride_qS) + (ds * stride_qD)
    k_offset = (heads * stride_kH) + (seqs * stride_kS) + (ds * stride_kD)
    cos_offset = (heads * stride_cosH) + (seqs * stride_cosS) + (ds * stride_cosD)
    sin_offset = (heads * stride_sinH) + (seqs * stride_sinS) + (ds * stride_sinD)

    # Load Q, K, Cos, Sin
    q = tl.load(Q_ptr + q_offset, mask=mask, other=0.0)
    k = tl.load(K_ptr + k_offset, mask=mask, other=0.0)
    cos = tl.load(Cos_ptr + cos_offset, mask=mask, other=0.0)
    sin = tl.load(Sin_ptr + sin_offset, mask=mask, other=0.0)

    # Split Q and K into two halves
    half = BLOCK_DMODEL // 2
    d_half_mask = ds < half
    q0 = tl.where(d_half_mask, q, 0.0)
    q1 = tl.where(~d_half_mask, q, 0.0)
    k0 = tl.where(d_half_mask, k, 0.0)
    k1 = tl.where(~d_half_mask, k, 0.0)

    # Same split logic for cos, sin
    cos0 = tl.where(d_half_mask, cos, 0.0)
    cos1 = tl.where(~d_half_mask, cos, 0.0)
    sin0 = tl.where(d_half_mask, sin, 0.0)
    sin1 = tl.where(~d_half_mask, sin, 0.0)

    # Apply rotary transformation
    out_q0 = q0 * cos0 - q1 * sin0
    out_q1 = q0 * sin1 + q1 * cos1
    out_k0 = k0 * cos0 - k1 * sin0
    out_k1 = k0 * sin1 + k1 * cos1

    # Merge halves back together
    q_rot = out_q0 + out_q1
    k_rot = out_k0 + out_k1

    # Store back
    tl.store(Q_ptr + q_offset, q_rot, mask=mask)
    tl.store(K_ptr + k_offset, k_rot, mask=mask)


def rotary_emb_fwd(Q, K, Cos, Sin):
    # Input validation
    assert Q.is_contiguous(), "Q must be contiguous"
    assert K.is_contiguous(), "K must be contiguous"
    assert Cos.is_contiguous(), "Cos must be contiguous"
    assert Sin.is_contiguous(), "Sin must be contiguous"
    assert Q.shape == K.shape, "Q and K must have the same shape"
    assert Q.shape == Cos.shape, "Q and Cos must have the same shape"
    assert Q.shape == Sin.shape, "Q and Sin must have the same shape"

    nheads, nseq, dmodel = Q.shape
    BLOCK_HEAD = 1
    BLOCK_SEQ = 64  # example block size for sequence dimension
    BLOCK_DMODEL = 64  # example block size for dmodel dimension

    # Strides
    stride_qH, stride_qS, stride_qD = Q.stride()
    stride_kH, stride_kS, stride_kD = K.stride()
    stride_cosH, stride_cosS, stride_cosD = Cos.stride()
    stride_sinH, stride_sinS, stride_sinD = Sin.stride()

    # Grid
    grid = (
        (nheads + BLOCK_HEAD - 1) // BLOCK_HEAD,
        (nseq + BLOCK_SEQ - 1) // BLOCK_SEQ
    )

    # Determine number of warps based on dimension size
    # (heuristic: concurrency = block_dim / 32)
    num_warps = 4
    if dmodel > 256:
        num_warps = 8

    # Kernel launch
    _rotary_kernel[grid](
        Q, K, Cos, Sin,
        nheads, nseq, dmodel,
        stride_qH, stride_qS, stride_qD,
        stride_kH, stride_kS, stride_kD,
        stride_cosH, stride_cosS, stride_cosD,
        stride_sinH, stride_sinS, stride_sinD,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps
    )
