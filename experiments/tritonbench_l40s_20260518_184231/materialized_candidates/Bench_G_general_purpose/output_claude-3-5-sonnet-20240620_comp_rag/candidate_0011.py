import triton
import triton.language as tl
import torch

ROPE_GROUP_SIZE = 4  # Can be tuned to 1, 2, 4, or 8

@triton.jit
def _rope_embedding(
    Q, Q_row_stride,
    cos, cos_row_stride, 
    sin, sin_row_stride,
    seqlen,
    head_dim: tl.constexpr,
    n_heads: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs for parallelization
    row_position = tl.program_id(0)  # Batch * sequence position
    group_head_position = tl.program_id(1)  # Group of heads
    
    # Create offsets for vectorized loading
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    # Load sin/cos values for current sequence position
    sin1 = tl.load(
        sin + (row_position % seqlen) * sin_row_stride + col_offsets,
        mask=mask, other=0
    )
    cos1 = tl.load(
        cos + (row_position % seqlen) * cos_row_stride + col_offsets,
        mask=mask, other=0
    )

    # Flip sign for backward pass
    if BACKWARD_PASS:
        sin1 = -sin1

    # Process heads in groups for better parallelization
    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min(head_start + ROPE_GROUP_SIZE, n_heads)

    # Apply RoPE to each head in the group
    for k in range(head_start, head_end):
        # Calculate offsets for the two halves of Q
        offs_q1 = row_position * Q_row_stride + k * head_dim + col_offsets
        offs_q2 = offs_q1 + half_head_dim

        # Load Q values and convert to sin1's dtype if needed
        Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
        Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)

        # Apply rotation: Q * cos + rotate_half(Q) * sin
        tl.store(Q + offs_q1, Q1 * cos1 - Q2 * sin1, mask=mask)
        tl.store(Q + offs_q2, Q2 * cos1 + Q1 * sin1, mask=mask)

class FastRoPEEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        # Prepare inputs
        cos, sin = cos.squeeze(), sin.squeeze()
        batch, seq_len, n_heads, head_dim = Q.shape
        Q = Q.view(batch * seq_len, n_heads * head_dim)
        
        # Calculate optimal settings
        BLOCK_SIZE = min(head_dim // 2, 128)  # Example simple heuristic
        num_warps = 4 if BLOCK_SIZE > 64 else 2
        
        # Calculate number of head groups
        n_groups = (n_heads + ROPE_GROUP_SIZE - 1) // ROPE_GROUP_SIZE

        # Launch kernel
        _rope_embedding[(batch * seq_len, n_groups)](
            Q, Q.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len,
            head_dim, n_heads,
            BACKWARD_PASS=False,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        # Save context for backward
        ctx.save_for_backward(cos, sin)
        ctx.dims = (batch, seq_len, n_heads, head_dim)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        
        return Q.view(batch, seq_len, n_heads, head_dim)

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        batch, seq_len, n_heads, head_dim = ctx.dims
        
        grad_output = grad_output.reshape(batch * seq_len, n_heads * head_dim)
        n_groups = (n_heads + ROPE_GROUP_SIZE - 1) // ROPE_GROUP_SIZE

        _rope_embedding[(batch * seq_len, n_groups)](
            grad_output, grad_output.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len, head_dim, n_heads,
            BACKWARD_PASS=True,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )

        return grad_output.view(*ctx.dims), None, None

# Convenience function
def apply_rope(Q, K, cos, sin):
    """Apply RoPE to query and key tensors"""
    Q = FastRoPEEmbedding.apply(Q.transpose(1, 2), cos, sin).transpose(1, 2)
    K = FastRoPEEmbedding.apply(K.transpose(1, 2), cos, sin).transpose(1, 2)
    return Q, K
