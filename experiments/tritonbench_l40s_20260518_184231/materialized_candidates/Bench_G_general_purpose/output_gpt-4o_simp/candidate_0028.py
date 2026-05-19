import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    X_ptr, cos_ptr, sin_ptr, out_ptr,
    n_heads, head_dim, seq_len,
    interleaved: tl.constexpr, conjugate: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Compute block indices
    pid = tl.program_id(0)
    bid = tl.program_id(1)

    # Compute position in sequence and head dimension
    seq_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    head_idx = bid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load input data
    if interleaved:
        # Interleaved layout
        x_idx = seq_idx[:, None] * n_heads * head_dim + head_idx[None, :]
    else:
        # Non-interleaved layout
        x_idx = bid * seq_len * head_dim + seq_idx[:, None] * head_dim + head_idx[None, :]

    # Load X, cos, sin values
    X = tl.load(X_ptr + x_idx, mask=(seq_idx[:, None] < seq_len) & (head_idx[None, :] < head_dim), other=0.0)
    cos = tl.load(cos_ptr + head_idx, mask=head_idx < head_dim, other=1.0)
    sin = tl.load(sin_ptr + head_idx, mask=head_idx < head_dim, other=0.0)

    # Apply rotary transformation
    X_rotated_real = X * cos - X * sin
    X_rotated_imag = X * sin + X * cos

    if conjugate:
        X_rotated_imag = -X_rotated_imag

    # Store result
    tl.store(out_ptr + x_idx, X_rotated_real + 1j * X_rotated_imag, mask=(seq_idx[:, None] < seq_len) & (head_idx[None, :] < head_dim))


def apply_rotary(X, cos, sin, interleaved=False, conjugate=False):
    # Determine input shapes
    batch_size, seq_len, n_heads, head_dim = X.shape

    # Allocate output tensor
    out = torch.empty_like(X, dtype=torch.complex64)

    # Define grid size
    grid = (triton.cdiv(seq_len, BLOCK_SIZE), n_heads)

    # Launch Triton kernel
    rotary_kernel[grid](
        X_ptr=X,
        cos_ptr=cos,
        sin_ptr=sin,
        out_ptr=out,
        n_heads=n_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        interleaved=interleaved,
        conjugate=conjugate,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
