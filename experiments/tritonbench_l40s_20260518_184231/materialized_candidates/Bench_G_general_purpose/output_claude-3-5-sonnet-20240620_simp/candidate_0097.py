:triton_kernel.py
import torch
import triton
import triton.language as tl

# Define block sizes as powers of 2 for efficient memory access
BLOCK_HEAD = 32  # Can be tuned based on head dimension size
BLOCK_DMODEL = 64  # Can be tuned based on model dimension size

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    # Pointers to tensors
    K_ptr, Out_ptr, DestLoc_ptr,
    # Dimensions
    batch_size, num_heads, head_dim,
    # Strides for memory access
    K_batch_stride, K_head_stride, K_dim_stride,
    Out_batch_stride, Out_head_stride, Out_dim_stride,
    # Meta-parameters
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate current batch index
    batch_idx = pid // ((num_heads + BLOCK_HEAD - 1) // BLOCK_HEAD)
    head_block_idx = pid % ((num_heads + BLOCK_HEAD - 1) // BLOCK_HEAD)
    
    # Calculate head range for this program instance
    head_start = head_block_idx * BLOCK_HEAD
    head_end = min(head_start + BLOCK_HEAD, num_heads)
    
    # Load destination index for current batch
    dest_idx = tl.load(DestLoc_ptr + batch_idx)
    
    # Handle dimensions
    dim_offsets = tl.arange(0, BLOCK_DMODEL)
    
    # Loop over heads assigned to this program instance
    for head_idx in range(head_start, head_end):
        # Calculate source and destination offsets
        K_offset = (batch_idx * K_batch_stride + 
                   head_idx * K_head_stride)
        Out_offset = (dest_idx * Out_batch_stride + 
                     head_idx * Out_head_stride)
        
        # Process elements in blocks of BLOCK_DMODEL
        for dim_start in range(0, head_dim, BLOCK_DMODEL):
            dim_end = min(dim_start + BLOCK_DMODEL, head_dim)
            dim_size = dim_end - dim_start
            
            # Create mask for valid elements
            mask = dim_offsets < dim_size
            
            # Load values from source tensor
            k_vals = tl.load(K_ptr + K_offset + (dim_start + dim_offsets) * K_dim_stride, 
                           mask=mask)
            
            # Store values to destination tensor
            tl.store(Out_ptr + Out_offset + (dim_start + dim_offsets) * Out_dim_stride,
                    k_vals, mask=mask)

def destindex_copy_kv(K: torch.Tensor, DestLoc: torch.Tensor, Out: torch.Tensor):
    """
