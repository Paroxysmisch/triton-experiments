# Example usage
Q = ...  # Input query tensor
K = ...  # Input key tensor
V = ...  # Input value tensor
B0 = ...  # Relative positional embedding tensor
acc = ...  # Accumulator tensor

sm_scale = ...  # Scaling factor for softmax
BLOCK_M = ...  # Block size for rows
BLOCK_N = ...  # Block size for columns
BLOCK_DMODEL = ...  # Block size for depth
SEQUENCE_LENGTH = ...  # Length of the sequence
HEADS = ...  # Number of attention heads
BATCH_SIZE = ...  # Batch size

# Launch the kernel
_attention_rel_h_rel_w_kernel_aligned_device(
    Q, K, V, B0, acc,
    Q_ptr, K_ptr, V_ptr, B0_ptr, acc_ptr,
    sm_scale,
    BLOCK_M, BLOCK_N, BLOCK_DMODEL,
    SEQUENCE_LENGTH, HEADS, BATCH_SIZE
)
