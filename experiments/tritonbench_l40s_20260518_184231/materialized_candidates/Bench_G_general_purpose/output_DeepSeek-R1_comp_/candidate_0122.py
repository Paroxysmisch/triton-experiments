import torch
import triton
import triton.language as tl

@triton.jit
def _rope_embedding(
    Q_ptr, Out_ptr, cos_ptr, sin_ptr,
    Q_row_stride, cos_row_stride, sin_row_stride,
    seqlen, head_dim, n_heads,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid

    # Compute pointers to the current row for Q, cos, sin
    Q_row_start = Q_ptr + row * Q_row_stride
    cos_row_start = cos_ptr + row * cos_row_stride
    sin_row_start = sin_ptr + row * sin_row_stride
    Out_row_start = Out_ptr + row * Q_row_stride

    # Process elements in the current row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < head_dim

    # Load data from Q, cos, sin
    q = tl.load(Q_row_start + offsets, mask=mask)
    cos = tl.load(cos_row_start + offsets, mask=mask)
    sin = tl.load(sin_row_start + offsets, mask=mask)

    # Compute rotated indices and signs for rotate_half
    half_dim = head_dim // 2
    rot_offsets = tl.where(offsets < half_dim, offsets + half_dim, offsets - half_dim)
    signs = tl.where(offsets < half_dim, -1.0, 1.0)

    # Load rotated Q values and apply sign
    q_rot = signs * tl.load(Q_row_start + rot_offsets, mask=mask)

    # Compute output element: Q * cos + rotate_half(Q) * sin
    out = q * cos + q_rot * sin

    # Store result
    tl.store(Out_row_start + offsets, out, mask=mask)

class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin, seqlen, n_heads, head_dim, BLOCK_SIZE):
        ctx.save_for_backward(cos, sin)
        ctx.head_dim = head_dim
        ctx.BLOCK_SIZE = BLOCK_SIZE

        # Reshape Q to 2D (batch*seqlen*n_heads, head_dim)
        batch_size = Q.size(0)
        Q_2D = Q.reshape(-1, head_dim)
        output = torch.empty_like(Q_2D)

        # Grid config: one kernel per row
        grid = (Q_2D.size(0),)
        _rope_embedding[grid](
            Q_2D, output, cos, sin,
            Q_2D.stride(0), cos.stride(0), sin.stride(0),
            seqlen, head_dim, n_heads,
            BACKWARD_PASS=False,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output.view_as(Q)

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        grad_output_2D = grad_output.reshape(-1, ctx.head_dim)
        grad_Q = torch.empty_like(grad_output_2D)

        grid = (grad_output_2D.size(0),)
        _rope_embedding[grid](
            grad_output_2D, grad_Q, cos, sin,
            grad_output_2D.stride(0), cos.stride(0), sin.stride(0),
            ctx.seqlen, ctx.n_heads, ctx.head_dim,
            BACKWARD_PASS=True,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )
        return grad_Q.view_as(grad_output), None, None, None, None, None, None

def fast_rope_embedding(Q, K, cos, sin, seqlen, n_heads, head_dim, BLOCK_SIZE=1024):
    # Ensure Q and K are in (batch, n_heads, seqlen, head_dim) format
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    
    Q = Fast_RoPE_Embedding.apply(Q, cos, sin, seqlen, n_heads, head_dim, BLOCK_SIZE)
    K = Fast_RoPE_Embedding.apply(K, cos, sin, seqlen, n_heads, head_dim, BLOCK_SIZE)
    
    # Transpose back to original shape (batch, seqlen, n_heads, head_dim)
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    return Q, K

# Q/K shape: (batch, seq_len, n_heads, head_dim)
# cos/sin shape: (seq_len, head_dim) -> expanded to (batch, seq_len, n_heads, head_dim)
Q_rot, K_rot = fast_rope_embedding(Q, K, cos, sin, seqlen, n_heads, head_dim)
