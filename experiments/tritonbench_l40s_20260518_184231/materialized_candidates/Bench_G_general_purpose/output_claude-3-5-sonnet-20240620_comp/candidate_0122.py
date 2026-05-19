import triton
import triton.language as tl
import torch

@triton.jit
def _rope_embedding_kernel(
    # Pointers to matrices
    Q_ptr, Q_row_stride,
    cos_ptr, cos_row_stride,
    sin_ptr, sin_row_stride,
    # Matrix dimensions
    seqlen, head_dim, n_heads,
    # Constants
    BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute the sequence position and head
    seq_id = pid // n_heads
    head_id = pid % n_heads
    
    if seq_id >= seqlen:
        return
        
    # Compute the starting offset for this sequence and head
    Q_offset = seq_id * Q_row_stride + head_id * head_dim
    cos_offset = seq_id * cos_row_stride
    sin_offset = seq_id * sin_row_stride
    
    # Load a block of elements
    for idx in range(0, head_dim, BLOCK_SIZE):
        # Create block mask
        mask = idx + tl.arange(0, BLOCK_SIZE) < head_dim
        
        # Load Q, cos, and sin values
        q = tl.load(Q_ptr + Q_offset + idx, mask=mask)
        cos_val = tl.load(cos_ptr + cos_offset + idx // 2, mask=mask)
        sin_val = tl.load(sin_ptr + sin_offset + idx // 2, mask=mask)
        
        # Compute rotated version of Q
        q_rot = tl.where(
            (idx + tl.arange(0, BLOCK_SIZE)) % 2 == 0,
            tl.concatenate([q[1::2], -q[::2]]),
            tl.concatenate([-q[1::2], q[::2]])
        )
        
        # Apply RoPE transformation
        output = q * cos_val + q_rot * sin_val
        
        # Store the result
        tl.store(Q_ptr + Q_offset + idx, output, mask=mask)


class FastRoPEEmbedding(torch.nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        
        # Initialize parameters
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        
        # Precompute position encodings
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, head_dim, 2) * (-math.log(base) / head_dim))
        
        # Create sin and cos tables
        self.register_buffer('cos_table', torch.cos(position * div_term))
        self.register_buffer('sin_table', torch.sin(position * div_term))
        
        # Constants for Triton kernel
        self.BLOCK_SIZE = 32
    
    def forward(self, q: torch.Tensor, k: torch.Tensor = None):
        # Get tensor dimensions
        batch_size, n_heads, seqlen, head_dim = q.shape
        
        # Ensure sequence length is within bounds
        assert seqlen <= self.max_seq_len, f"Sequence length {seqlen} exceeds maximum {self.max_seq_len}"
        
        # Reshape tensors for Triton kernel
        q = q.reshape(-1, head_dim)
        
        # Launch Triton kernel for query
        grid = (seqlen * n_heads * batch_size,)
        _rope_embedding_kernel[grid](
            q, q.stride(0),
            self.cos_table, self.cos_table.stride(0),
            self.sin_table, self.sin_table.stride(0),
            seqlen, head_dim, n_heads,
            self.BLOCK_SIZE
        )
        
        # Process key if provided
        if k is not None:
            k = k.reshape(-1, head_dim)
            _rope_embedding_kernel[grid](
                k, k.stride(0),
                self.cos_table, self.cos_table.stride(0),
                self.sin_table, self.sin_table.stride(0),
                seqlen, head_dim, n_heads,
                self.BLOCK_SIZE
            )
            k = k.reshape(batch_size, n_heads, seqlen, head_dim)
        
        # Reshape query back to original dimensions
        q = q.reshape(batch_size, n_heads, seqlen, head_dim)
        
        return (q, k) if k is not None else q
