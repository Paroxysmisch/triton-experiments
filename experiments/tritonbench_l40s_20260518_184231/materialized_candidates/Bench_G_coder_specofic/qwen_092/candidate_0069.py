# Example usage (pseudo-code)
Q = ...  # Query tensor
K = ...  # Key tensor
V = ...  # Value tensor
Out = ...  # Output tensor
seqlens = ...  # Sequence lengths
block_indices = ...  # Block indices
block_counts = ...  # Block counts
column_indices = ...  # Column indices
column_counts = ...  # Column counts
sm_scale = ...  # Scaling factor for dot products
qk_scale = ...  # Scaling factor for query

_triton_mixed_sparse_attention(
    Q, K, V, Out, seqlens, block_indices, block_counts, column_indices, column_counts, sm_scale, qk_scale,
    BLOCK_SIZE, SEQUENCE_LENGTH, HEAD_SIZE, COLUMN_SIZE
)
