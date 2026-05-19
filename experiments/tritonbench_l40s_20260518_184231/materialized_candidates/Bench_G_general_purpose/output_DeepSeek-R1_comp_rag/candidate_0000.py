rotated = Q * cos + rotate_half(Q) * sin

import triton
import triton.language as tl

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
    ROPE_GROUP_SIZE: tl.constexpr,
):
    # Parallelize over rows and head groups
    row_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_dim = head_dim // 2
    mask = col_offsets < half_dim
    
    # Load position-dependent cos/sin values
    pos = row_idx % seqlen
    sin_val = tl.load(sin + pos * sin_row_stride + col_offsets, mask=mask)
    cos_val = tl.load(cos + pos * cos_row_stride + col_offsets, mask=mask)
    
    if BACKWARD_PASS:
        sin_val = -sin_val

    # Process head group
    head_start = group_idx * ROPE_GROUP_SIZE
    heads = tl.arange(head_start, head_start + ROPE_GROUP_SIZE)
    mask_heads = heads < n_heads
    
    for head in tl.static_range(ROPE_GROUP_SIZE):
        if mask_heads[head]:
            # Calculate memory offsets
            head_offset = (head_start + head) * head_dim
            q_ptr = Q + row_idx * Q_row_stride + head_offset
            
            # Load and rotate Q values
            q1 = tl.load(q_ptr + col_offsets, mask=mask)
            q2 = tl.load(q_ptr + half_dim + col_offsets, mask=mask)
            
            # Apply rotation
            new_q1 = q1 * cos_val - q2 * sin_val
            new_q2 = q2 * cos_val + q1 * sin_val
            
            # Store results
            tl.store(q_ptr + col_offsets, new_q1, mask=mask)
            tl.store(q_ptr + half_dim + col_offsets, new_q2, mask=mask)

class RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        # Reshape for kernel: [batch*seqlen, n_heads*head_dim]
        orig_shape = Q.shape
        batch, seqlen, n_heads, head_dim = orig_shape
        Q = Q.reshape(-1, n_heads * head_dim)
        
        # Kernel configuration
        BLOCK_SIZE, num_warps = calculate_settings(head_dim//2)
        ROPE_GROUP_SIZE = 4
        n_groups = triton.cdiv(n_heads, ROPE_GROUP_SIZE)
        
        # Launch kernel
        grid = (Q.shape[0], n_groups)
        _rope_embedding[grid](
            Q, Q.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seqlen,
            head_dim, n_heads,
            False, BLOCK_SIZE, ROPE_GROUP_SIZE,
            num_warps=num_warps
        )
        
        ctx.save_for_backward(cos, sin)
        ctx.dims = (seqlen, head_dim, n_heads, BLOCK_SIZE, ROPE_GROUP_SIZE)
        return Q.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        seqlen, head_dim, n_heads, BLOCK_SIZE, ROPE_GROUP_SIZE = ctx.dims
        
        # Reshape gradient
        grad = grad_output.reshape(-1, n_heads * head_dim)
        n_groups = triton.cdiv(n_heads, ROPE_GROUP_SIZE)
        
        # Launch kernel with inverse sign
        grid = (grad.shape[0], n_groups)
        _rope_embedding[grid](
            grad, grad.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seqlen,
            head_dim, n_heads,
            True, BLOCK_SIZE, ROPE_GROUP_SIZE,
            num_warps=ctx.num_warps
        )
        
        return grad.reshape(grad_output.shape), None, None

def calculate_settings(n):
    max_fused_size = 1024  # Adjust based on GPU architecture
    block_size = 1 << (n - 1).bit_length()
    
    if block_size > max_fused_size:
        raise ValueError(f"Head dimension {n*2} too large for RoPE kernel")
    
    num_warps = 4 if block_size >= 512 else \
                 2 if block_size >= 256 else \
                 1
    return block_size, num_warps

def apply_rope(Q, K, cos, sin):
    Q = RoPEFunction.apply(Q.transpose(1, 2), cos, sin).transpose(1, 2)
    K = RoPEFunction.apply(K.transpose(1, 2), cos, sin).transpose(1, 2)
    return Q, K
