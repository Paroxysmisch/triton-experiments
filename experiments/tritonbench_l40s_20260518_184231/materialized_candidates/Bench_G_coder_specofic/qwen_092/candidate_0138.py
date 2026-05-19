import triton
import triton.language as tl
import torch

# Define the block sizes and other parameters
BLOCK_M = 16
BLOCK_N = 16
BLOCK_D = 64
NUM_D_BLOCKS = 1
num_heads = 4
num_kv_heads = 4

# Create example tensors
Q = torch.randn(32, 16, 64, device='cuda')
K = torch.randn(32, 16, 64, device='cuda')
V = torch.randn(32, 16, 64, device='cuda')
out = torch.zeros(32, 16, 64, device='cuda')

# Create example layout data (CSR format)
layout_csr_row_indices = torch.tensor([0, 16, 32], device='cuda')
layout_csr_col_indices = torch.tensor([0, 16, 32], device='cuda')

# Launch the wrapper function
block_sparse_attention_wrapper[triton.cdiv(Q.shape[0], BLOCK_M), triton.cdiv(Q.shape[1], BLOCK_N)](
    Q, K, V, out,
    layout_csr_row_indices, layout_csr_col_indices,
    softmax_scale=1.0,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    num_heads=num_heads, num_kv_heads=num_kv_heads,
    NUM_D_BLOCKS=NUM_D_BLOCKS
)

# Print the result
print(out)
