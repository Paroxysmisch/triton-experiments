import triton
import triton.language as tl
import torch

# Define the block constants
BLOCK_HEAD = 1
BLOCK_SEQ = 1
BLOCK_DMODEL = 16

# Create example tensors
Q = torch.randn(BLOCK_HEAD * BLOCK_SEQ, BLOCK_DMODEL * 2, dtype=torch.float32, device='cuda')
K = torch.randn(BLOCK_HEAD * BLOCK_SEQ, BLOCK_DMODEL * 2, dtype=torch.float32, device='cuda')
Cos = torch.randn(BLOCK_HEAD * BLOCK_DMODEL, dtype=torch.float32, device='cuda')
Sin = torch.randn(BLOCK_HEAD * BLOCK_DMODEL, dtype=torch.float32, device='cuda')

# Convert tensors to Triton tensors
Q_triton = triton.tensor(Q)
K_triton = triton.tensor(K)
Cos_triton = triton.tensor(Cos)
Sin_triton = triton.tensor(Sin)

# Launch the kernel
rotary_emb_fwd(Q_triton, K_triton, Cos_triton, Sin_triton, BLOCK_HEAD, BLOCK_SEQ, BLOCK_DMODEL)

# Convert the results back to PyTorch tensors if needed
Q_result = Q_triton.to(torch.float32)
K_result = K_triton.to(torch.float32)
