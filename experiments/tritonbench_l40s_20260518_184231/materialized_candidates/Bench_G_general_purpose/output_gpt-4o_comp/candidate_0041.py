import triton
import triton.language as tl
import torch

@triton.jit
def _triton_rope(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    q_stride, k_stride, cos_stride, sin_stride,
    head_dim, num_heads, seq_len, batch_size,
    BLOCK_SIZE: tl.constexpr, BACKWARD_PASS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    # Compute batch, sequence, and head indices
    batch_id = pid // seq_len
    seq_id = pid % seq_len
    
    # Compute pointers for the start of each batch/sequence
    q_offset = batch_id * q_stride[0] + seq_id * q_stride[1]
    k_offset = batch_id * k_stride[0] + seq_id * k_stride[1]
    
    # Load slices of q and k
    q = tl.load(q_ptr + q_offset + tl.arange(0, BLOCK_SIZE))
    k = tl.load(k_ptr + k_offset + tl.arange(0, BLOCK_SIZE))
    
    # Load cosine and sine values
    cos = tl.load(cos_ptr + tl.arange(0, BLOCK_SIZE))
    sin = tl.load(sin_ptr + tl.arange(0, BLOCK_SIZE))
    
    # Apply rotary transformation
    if BACKWARD_PASS:
        q_rotated = q * cos + k * sin
        k_rotated = k * cos - q * sin
    else:
        q_rotated = q * cos - k * sin
        k_rotated = k * cos + q * sin
    
    # Store the transformed slices back
    tl.store(q_ptr + q_offset + tl.arange(0, BLOCK_SIZE), q_rotated)
    tl.store(k_ptr + k_offset + tl.arange(0, BLOCK_SIZE), k_rotated)

def rope_forward(q, k, cos, sin, backward_pass=False):
    # Ensure input tensors are contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    
    # Get dimensions
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Prepare strides
    q_stride = q.stride()
    k_stride = k.stride()
    cos_stride = cos.stride()
    sin_stride = sin.stride()
    
    # Determine block size
    BLOCK_SIZE = head_dim
    
    # Launch the kernel
    grid = (batch_size * seq_len,)
    _triton_rope[grid](
        q, k, cos, sin,
        q_stride, k_stride, cos_stride, sin_stride,
        head_dim, num_heads, seq_len, batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
        BACKWARD_PASS=backward_pass
    )
    
    return q, k

# Example usage:
batch_size = 2
seq_len = 128
num_heads = 8
head_dim = 64

q = torch.randn((batch_size, seq_len, num_heads, head_dim), device='cuda')
k = torch.randn((batch_size, seq_len, num_heads, head_dim), device='cuda')
cos = torch.cos(torch.linspace(0, 3.14, head_dim, device='cuda'))
sin = torch.sin(torch.linspace(0, 3.14, head_dim, device='cuda'))

q_transformed, k_transformed = rope_forward(q, k, cos, sin)
