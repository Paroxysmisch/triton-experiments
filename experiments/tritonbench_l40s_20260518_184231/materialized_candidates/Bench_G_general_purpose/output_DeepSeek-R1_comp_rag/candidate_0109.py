import triton
import triton.language as tl
import torch
from torch.cuda.amp import custom_bwd, custom_fwd

ROPE_GROUP_SIZE = 4  # Tune based on hardware characteristics

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
    row_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_dim = head_dim // 2
    
    # Position-aware trigonometric loading with sequence length modulation
    pos = row_idx % seqlen
    mask = col_offsets < half_dim
    cos_vals = tl.load(cos + pos*cos_row_stride + col_offsets, mask=mask, other=0)
    sin_vals = tl.load(sin + pos*sin_row_stride + col_offsets, mask=mask, other=0)
    
    # Gradient sign adjustment
    if BACKWARD_PASS:
        sin_vals = -sin_vals

    # Process ROPE_GROUP_SIZE heads per block for better locality
    head_start = group_idx * ROPE_GROUP_SIZE
    head_end = min(head_start + ROPE_GROUP_SIZE, n_heads)

    for head in range(head_start, head_end):
        # Calculate memory offsets for head components
        base = row_idx * Q_row_stride + head * head_dim
        q1_ptr = Q + base + col_offsets
        q2_ptr = Q + base + half_dim + col_offsets
        
        # Load and transform components
        q1 = tl.load(q1_ptr, mask=mask, other=0).to(sin_vals.dtype)
        q2 = tl.load(q2_ptr, mask=mask, other=0).to(sin_vals.dtype)
        
        # Apply rotary transformation
        tl.store(q1_ptr, q1 * cos_vals - q2 * sin_vals, mask=mask)
        tl.store(q2_ptr, q2 * cos_vals + q1 * sin_vals, mask=mask)

class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, Q, cos, sin):
        batch_size, seq_len, n_heads, head_dim = Q.shape
        
        # Flatten batch and sequence dimensions for kernel efficiency
        Q_flat = Q.reshape(-1, n_heads * head_dim)
        n_rows, n_cols = Q_flat.shape
        
        # Kernel configuration
        BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)
        BLOCK_SIZE = max(BLOCK_SIZE, 16)
        num_warps = 4 if BLOCK_SIZE <= 128 else 8
        
        # Group heads for coalesced memory access
        n_groups = (n_heads + ROPE_GROUP_SIZE - 1) // ROPE_GROUP_SIZE

        # Launch kernel
        _rope_embedding[(n_rows, n_groups)](
            Q_flat, Q_flat.stride(0),
            cos.contiguous(), cos.stride(0),
            sin.contiguous(), sin.stride(0),
            seq_len,
            head_dim, n_heads,
            BACKWARD_PASS=False,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        
        ctx.save_for_backward(cos, sin)
        ctx.dimensions = (seq_len, head_dim, n_heads, BLOCK_SIZE, num_warps)
        return Q_flat.view_as(Q)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        seq_len, head_dim, n_heads, BLOCK_SIZE, num_warps = ctx.dimensions
        
        # Prepare gradient tensor
        grad_flat = grad_output.reshape(-1, n_heads * head_dim)
        n_rows = grad_flat.size(0)
        n_groups = (n_heads + ROPE_GROUP_SIZE - 1) // ROPE_GROUP_SIZE

        # Apply inverse transformation using stored parameters
        _rope_embedding[(n_rows, n_groups)](
            grad_flat, grad_flat.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len,
            head_dim, n_heads,
            BACKWARD_PASS=True,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        
        return grad_output, None, None

def fast_rope_embedding(Q, K, cos, sin):
    # Ensure tensor dimensions: [batch, seq_len, heads, dim]
    Q = Fast_RoPE_Embedding.apply(Q, cos, sin)
    K = Fast_RoPE_Embedding.apply(K, cos, sin)
    return Q, K
