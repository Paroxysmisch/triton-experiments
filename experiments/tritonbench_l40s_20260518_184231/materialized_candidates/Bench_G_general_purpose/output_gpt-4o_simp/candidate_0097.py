import triton
import triton.language as tl

# Define the kernel
@triton.jit
def _fwd_kernel_destindex_copy_kv(K, Out, DestLoc, BLOCK_HEAD: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    # Get the program id
    pid = tl.program_id(0)
    
    # Compute the head and depth offsets for the current block
    head_offset = pid // BLOCK_DMODEL
    depth_offset = pid % BLOCK_DMODEL
    
    # Load the destination index for the current block
    dest_index = tl.load(DestLoc + head_offset)
    
    # Compute the base pointers for K and Out
    K_ptrs = K + dest_index * K.stride(0) + head_offset * K.stride(1) + depth_offset
    Out_ptrs = Out + pid * Out.stride(1)
    
    # Load data from K and store it into Out
    k_data = tl.load(K_ptrs)
    tl.store(Out_ptrs, k_data)

# Define the wrapper function
def destindex_copy_kv(K, Out, DestLoc, BLOCK_HEAD, BLOCK_DMODEL):
    # Determine the grid size
    grid = (K.shape[1] * K.shape[2] + BLOCK_DMODEL - 1) // BLOCK_DMODEL
    
    # Launch the kernel
    _fwd_kernel_destindex_copy_kv[grid](
        K, Out, DestLoc,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_DMODEL=BLOCK_DMODEL
    )

# Example usage
# Assuming K is a 3D tensor of shape [batch, head, depth]
# Out is a tensor of the same shape as K
# DestLoc is a 1D tensor of length equal to the number of heads
import torch

batch_size = 32
num_heads = 8
depth = 64

K = torch.randn((batch_size, num_heads, depth), device='cuda')
Out = torch.empty_like(K)
DestLoc = torch.randint(0, batch_size, (num_heads,), device='cuda')

BLOCK_HEAD = 1  # Must be a power of two
BLOCK_DMODEL = 64  # Must be a power of two

destindex_copy_kv(K, Out, DestLoc, BLOCK_HEAD, BLOCK_DMODEL)
