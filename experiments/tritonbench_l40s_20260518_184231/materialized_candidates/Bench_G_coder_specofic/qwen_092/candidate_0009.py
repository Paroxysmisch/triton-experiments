import triton
import triton.language as tl

# Constants
MAX_FUSED_SIZE = 256
ROPE_GROUP_SIZE = 1

@triton.jit
def calculate_settings(n):
    BLOCK_SIZE = 1
    num_warps = 1
    while BLOCK_SIZE < n and BLOCK_SIZE <= MAX_FUSED_SIZE:
        BLOCK_SIZE *= 2
        num_warps = min(num_warps * 2, 8)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Block size {BLOCK_SIZE} exceeds the maximum allowed size {MAX_FUSED_SIZE}")
    return BLOCK_SIZE, num_warps

@triton.jit
def _rope_embedding(Q, Q_row_stride, cos, cos_row_stride, sin, sin_row_stride, seqlen, head_dim, n_heads, BACKWARD_PASS, BLOCK_SIZE, ROPE_GROUP_SIZE):
    pid = tl.program_id(axis=0)
    num_blocks = tl.cdiv(seqlen, BLOCK_SIZE)
    grid_size = num_blocks * n_heads
    x = pid % BLOCK_SIZE
    y = pid // BLOCK_SIZE
    row = y // head_dim
    col = y % head_dim

    # Load Q, cos, sin
    q = tl.load(Q + row * Q_row_stride + col, mask=x < BLOCK_SIZE, other=0.0)
    c = tl.load(cos + row * cos_row_stride + col, mask=x < BLOCK_SIZE, other=0.0)
    s = tl.load(sin + row * sin_row_stride + col, mask=x < BLOCK_SIZE, other=0.0)

    # Compute RoPE embeddings
    if BACKWARD_PASS:
        q = q * c - q[::-1] * s
    else:
        q = q * c + q[::-1] * s

    # Store the result
    tl.store(Q + row * Q_row_stride + col, q, mask=x < BLOCK_SIZE)

@triton.jit
def _rope_embedding_forward_impl(Q, cos, sin):
    seqlen, head_dim, n_heads = Q.shape[0], Q.shape[2], Q.shape[3]
    BLOCK_SIZE, num_warps = calculate_settings(seqlen)
    n_groups = n_heads // ROPE_GROUP_SIZE
    grid_size = n_groups * BLOCK_SIZE
    Q = Q.to(triton.float32)
    cos = cos.to(triton.float32)
    sin = sin.to(triton.float32)
    _rope_embedding[grid_size, num_warps](Q, Q.stride(0), cos, cos.stride(0), sin, sin.stride(0), seqlen, head_dim, n_heads, False, BLOCK_SIZE, ROPE_GROUP_SIZE)

@triton.jit
def _rope_embedding_backward_impl(dY, cos, sin, n_groups, BLOCK_SIZE, num_warps):
    seqlen, head_dim, n_heads = dY.shape[0], dY.shape[2], dY.shape[3]
    dY = dY.to(triton.float32)
    cos = cos.to(triton.float32)
    sin = sin.to(triton.float32)
    _rope_embedding[grid_size, num_warps](dY, dY.stride(0), cos, cos.stride(0), sin, sin.stride(0), seqlen, head_dim, n_heads, True, BLOCK_SIZE, ROPE_GROUP_SIZE)
