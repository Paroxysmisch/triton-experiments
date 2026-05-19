import triton
import triton.language as tl
import torch
from .utils import calculate_settings

# Define constant for grouping heads
ROPE_GROUP_SIZE = 4

@triton.jit
def _rope_embedding(
    Q, Q_row_stride,           # Query matrix and its row stride
    cos, cos_row_stride,       # Cosine values and stride
    sin, sin_row_stride,       # Sine values and stride
    seqlen,                    # Sequence length
    head_dim: tl.constexpr,    # Dimension of each attention head
    n_heads: tl.constexpr,     # Number of attention heads
    BACKWARD_PASS: tl.constexpr,# Flag for backward pass
    BLOCK_SIZE: tl.constexpr,  # Block size for parallel computation
):
    # Get thread indices
    row_position = tl.program_id(0)  # Position in sequence
    group_head_position = tl.program_id(1)  # Position in head groups
    
    # Create offset array for parallel processing
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    # Load sine and cosine values for current position
    sin1 = tl.load(sin + (row_position % seqlen)*sin_row_stride + 
                  half_head_dim*0 + col_offsets, mask=mask, other=0)
    cos1 = tl.load(cos + (row_position % seqlen)*cos_row_stride + 
                  half_head_dim*0 + col_offsets, mask=mask, other=0)

    # Flip sign for backward pass
    if BACKWARD_PASS:
        sin1 = -sin1

    # Process heads in groups
    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min((head_start + ROPE_GROUP_SIZE), n_heads)

    # Apply RoPE to each head in the group
    for k in range(head_start, head_end):
        # Calculate offsets for first and second half of dimensions
        offs_q1 = row_position * Q_row_stride + k * head_dim + col_offsets
        offs_q2 = offs_q1 + half_head_dim

        # Load query values
        Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
        Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)

        # Apply rotation: Q * cos + rotate_half(Q) * sin
        tl.store(Q + offs_q1, Q1*cos1 - Q2*sin1, mask=mask)
        tl.store(Q + offs_q2, Q2*cos1 + Q1*sin1, mask=mask)

class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        # Prepare input tensors
        cos, sin = cos.squeeze(), sin.squeeze()
        batch, seq_len, n_heads, head_dim = Q.shape
        Q = Q.view(batch*seq_len, n_heads*head_dim)
        n_rows, n_cols = Q.shape
        assert(seq_len <= cos.shape[0])

        # Calculate kernel launch configuration
        BLOCK_SIZE, num_warps = calculate_settings(head_dim//2)
        
        # Calculate number of head groups
        div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
        n_groups = div + (mod != 0)

        # Launch kernel
        _rope_embedding[(n_rows, n_groups,)](
            Q, Q.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len,
            head_dim, n_heads,
            BACKWARD_PASS=False,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        # Save context for backward pass
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.n_groups = n_groups
        ctx.cos = cos
        ctx.sin = sin
        
        return Q.view(batch, seq_len, n_heads, head_dim)

    @staticmethod
    def backward(ctx, dY):
        # Prepare gradients
        batch, seq_len, n_heads, head_dim = dY.shape
        dY = dY.reshape(batch*seq_len, n_heads*head_dim)
        n_rows, n_cols = dY.shape

        # Launch kernel for backward pass
        _rope_embedding[(n_rows, ctx.n_groups,)](
            dY, dY.stride(0),
            ctx.cos, ctx.cos.stride(0),
            ctx.sin, ctx.sin.stride(0),
            seq_len, head_dim, n_heads,
            BACKWARD_PASS=True,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )
        
        return dY.view(batch, seq_len, n_heads, head_dim), None, None

def fast_rope_embedding(Q, K, cos, sin):
    """Apply RoPE to query and key matrices"""
    Q = Fast_RoPE_Embedding.apply(Q.transpose(1, 2), cos, sin).transpose(1, 2)
    K = Fast_RoPE_Embedding.apply(K.transpose(1, 2), cos, sin).transpose(1, 2)
    return Q, K
