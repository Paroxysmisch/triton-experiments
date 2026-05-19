import torch
import triton
import triton.language as tl


@triton.jit
def _triton_rope(
    Q_PTR,  # pointer to Q
    K_PTR,  # pointer to K
    COS_PTR,
    SIN_PTR,
    Q_STRIDE_BATCH,
    Q_STRIDE_SEQLEN,
    Q_STRIDE_HEAD,
    Q_STRIDE_DIM,
    K_STRIDE_BATCH,
    K_STRIDE_SEQLEN,
    K_STRIDE_HEAD,
    K_STRIDE_DIM,
    COS_STRIDE,
    SIN_STRIDE,
    BATCH_SIZE,
    SEQ_LEN,
    HEAD_COUNT,
    HEAD_DIM,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    # Each pid_m corresponds to a set of rows within [batch_size * seq_len].
    total_rows = BATCH_SIZE * SEQ_LEN
    row_start = pid_m * BLOCK_SIZE
    # If we're past total rows, return immediately.
    if row_start >= total_rows:
        return

    # Create a range [row_start, row_start + BLOCK_SIZE).
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds.
    mask_m = offsets < total_rows
    # batch_id, seq_id from combined row index.
    batch_id = offsets // SEQ_LEN
    seq_id = offsets % SEQ_LEN

    # Each program_id(axis=1) handles a head index.
    if pid_h >= HEAD_COUNT:
        return

    # Compute pointer offsets for Q and K.
    q_offset = batch_id * Q_STRIDE_BATCH + seq_id * Q_STRIDE_SEQLEN + pid_h * Q_STRIDE_HEAD
    k_offset = batch_id * K_STRIDE_BATCH + seq_id * K_STRIDE_SEQLEN + pid_h * K_STRIDE_HEAD

    # Load the cosine and sine for the specific seq_id.
    # We assume COS_PTR and SIN_PTR each store [SEQ_LEN, HEAD_DIM/2] if needed, or enough for HEAD_DIM.
    # For each dimension d in [0..HEAD_DIM), load cos, sin, apply rotation to Q, K.
    d = tl.arange(0, BLOCK_SIZE)
    # We'll loop with a step of BLOCK_SIZE to handle HEAD_DIM if HEAD_DIM > BLOCK_SIZE.
    # This example is simplified: we only show one block iteration for clarity.

    # Limit HEAD_DIM to the block size; if HEAD_DIM > BLOCK_SIZE, multiple launches or a loop are needed.
    # For brevity, only a single pass:
    head_dim_range = tl.arange(0, BLOCK_SIZE)
    dim_mask = head_dim_range < HEAD_DIM

    q_ptr = Q_PTR + q_offset[:, None] + head_dim_range[None, :] * Q_STRIDE_DIM
    k_ptr = K_PTR + k_offset[:, None] + head_dim_range[None, :] * K_STRIDE_DIM

    cos_ptr = COS_PTR + seq_id[:, None] * COS_STRIDE + head_dim_range[None, :] // 2
    sin_ptr = SIN_PTR + seq_id[:, None] * SIN_STRIDE + head_dim_range[None, :] // 2

    # Load Q and K; mask with valid rows & dims.
    q_val = tl.load(q_ptr, mask=(mask_m[:, None] & dim_mask[None, :]), other=0.0)
    k_val = tl.load(k_ptr, mask=(mask_m[:, None] & dim_mask[None, :]), other=0.0)

    cos_val = tl.load(cos_ptr, mask=(mask_m[:, None] & dim_mask[None, :]), other=1.0)
    sin_val = tl.load(sin_ptr, mask=(mask_m[:, None] & dim_mask[None, :]), other=0.0)

    # Even dims vs odd dims for standard rope rotation in interleaved fashion.
    # For dimension i, we treat pairs (2j, 2j+1).
    # Q Even index = q_val, Q Odd index = q_val, then rotate with cos_val, sin_val.
    # If BACKWARD_PASS, we flip sign of sin.
    sign = -1.0 if BACKWARD_PASS else 1.0
    is_even = (head_dim_range % 2) == 0
    # Q rotation
    q_even = tl.where(is_even[None, :], q_val, 0.)
    q_odd = tl.where(~is_even[None, :], q_val, 0.)
    rotated_q = tl.where(
        is_even[None, :],
        q_even * cos_val - q_odd * (sign * sin_val),
        q_odd * cos_val + q_even * (sign * sin_val),
    )
    # K rotation
    k_even = tl.where(is_even[None, :], k_val, 0.)
    k_odd = tl.where(~is_even[None, :], k_val, 0.)
    rotated_k = tl.where(
        is_even[None, :],
        k_even * cos_val - k_odd * (sign * sin_val),
        k_odd * cos_val + k_even * (sign * sin_val),
    )

    # Store back
    tl.store(q_ptr, rotated_q, mask=(mask_m[:, None] & dim_mask[None, :]))
    tl.store(k_ptr, rotated_k, mask=(mask_m[:, None] & dim_mask[None, :]))


def rope_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    backward_pass: bool = False,
):
    assert q.shape == k.shape, "Q and K must have the same shape"
    batch_size, seq_len, num_heads, head_dim = q.shape
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    # Grid: we parallelize over [batch_size * seq_len, num_heads]
    grid = (triton.cdiv(batch_size * seq_len, 32), num_heads)
    block_size = 32

    _triton_rope[grid](
        q,  # Q_PTR
        k,  # K_PTR
        cos,
        sin,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        cos.stride(0),  # COS_STRIDE
        sin.stride(0),  # SIN_STRIDE
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        backward_pass,
        block_size,
    )
    return q, k, cos, sin


@triton.jit
def rotary_kernel(
    OUT,  # Pointers to matrices
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,
