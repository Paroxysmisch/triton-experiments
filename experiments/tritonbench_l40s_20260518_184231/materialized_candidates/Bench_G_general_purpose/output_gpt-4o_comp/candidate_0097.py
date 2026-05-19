import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(K, Out, DestLoc, BLOCK_HEAD: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    pid = tl.program_id(0)  # unique program id for each sequence element
    cur_index = pid  # sequence index
    
    # Compute offsets for accessing head and depth dimensions
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Compute the base pointers for the current sequence element
    dest_loc = tl.load(DestLoc + cur_index)
    
    # Base pointers for K and Out
    K_ptrs = K + dest_loc * BLOCK_HEAD * BLOCK_DMODEL + offs_h[:, None] * BLOCK_DMODEL + offs_d[None, :]
    Out_ptrs = Out + cur_index * BLOCK_HEAD * BLOCK_DMODEL + offs_h[:, None] * BLOCK_DMODEL + offs_d[None, :]
    
    # Load data from K and store it to Out
    k_vals = tl.load(K_ptrs)
    tl.store(Out_ptrs, k_vals)

def destindex_copy_kv(K, DestLoc, BLOCK_HEAD=16, BLOCK_DMODEL=16):
    # Ensure that K and DestLoc are on the GPU
    assert K.is_cuda and DestLoc.is_cuda, "Tensors must be on GPU"
    
    # Dimensions
    B, H, D = K.shape  # Batch size, head number, head dimension
    seq_length = DestLoc.shape[0]
    
    # Output tensor
    Out = torch.empty((seq_length, H, D), device=K.device, dtype=K.dtype)
    
    # Define grid size
    grid = (seq_length,)
    
    # Launch the Triton kernel
    _fwd_kernel_destindex_copy_kv[grid](
        K, Out, DestLoc,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_DMODEL=BLOCK_DMODEL
    )
    
    return Out

# Example usage
B, H, D = 64, 16, 64  # Example dimensions
K = torch.randn(B, H, D, device='cuda')
DestLoc = torch.randint(0, B, (B,), device='cuda', dtype=torch.int32)

# Execute the function
Out = destindex_copy_kv(K, DestLoc)
