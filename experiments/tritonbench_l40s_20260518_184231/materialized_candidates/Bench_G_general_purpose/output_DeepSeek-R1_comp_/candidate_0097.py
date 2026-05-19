import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K,  # Pointer to the input tensor
    Out,  # Pointer to the output tensor
    DestLoc,  # Pointer to destination indices tensor
    K_batch_stride, K_head_stride, K_dim_stride,  # Strides for K
    Out_batch_stride, Out_head_stride, Out_dim_stride,  # Strides for Out
    seq_len,  # Sequence length (size of DestLoc)
    num_heads,  # Number of attention heads
    head_dim,  # Dimension per head
    BLOCK_HEAD: tl.constexpr,  # Block size for heads (power of two)
    BLOCK_DMODEL: tl.constexpr,  # Block size for dimensions (power of two)
):
    pid = tl.program_id(0)
    
    # Calculate number of blocks per sequence element
    num_blocks_head = tl.cdiv(num_heads, BLOCK_HEAD)
    num_blocks_dmodel = tl.cdiv(head_dim, BLOCK_DMODEL)
    blocks_per_sequence = num_blocks_head * num_blocks_dmodel
    
    # Decompose program ID into components
    cur_index = pid // blocks_per_sequence
    block_id = pid % blocks_per_sequence
    head_block = block_id // num_blocks_dmodel
    d_block = block_id % num_blocks_dmodel

    # Boundary check for sequence length
    if cur_index >= seq_len:
        return
    
    # Load destination batch index for current sequence position
    dest_batch = tl.load(DestLoc + cur_index)
    
    # Calculate offsets for heads and dimensions
    off_h = head_block * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    off_d = d_block * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)
    
    # Create masks for valid heads/dimensions
    mask_h = off_h < num_heads
    mask_d = off_d < head_dim
    mask = mask_h[:, None] & mask_d[None, :]
    
    # Calculate pointers for current block in K
    k_ptrs = (
        K + cur_index * K_batch_stride + 
        off_h[:, None] * K_head_stride + 
        off_d[None, :] * K_dim_stride
    )
    
    # Calculate pointers for destination block in Out
    out_ptrs = (
        Out + dest_batch * Out_batch_stride +
        off_h[:, None] * Out_head_stride +
        off_d[None, :] * Out_dim_stride
    )
    
    # Load and store data with masking
    data = tl.load(k_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, data, mask=mask)


def destindex_copy_kv(K: torch.Tensor, DestLoc: torch.Tensor, Out: torch.Tensor):
    # Validate tensor dimensions
    assert K.dim() == 3, "Input tensor K must be 3D"
    assert Out.dim() == 3, "Output tensor Out must be 3D"
    assert K.size(1) == Out.size(1), "Head count mismatch"
    assert K.size(2) == Out.size(2), "Head dimension mismatch"
    
    seq_len = DestLoc.size(0)
    num_heads, head_dim = K.shape[1], K.shape[2]
    
    # Calculate block sizes as next power of two
    BLOCK_HEAD = triton.next_power_of_two(num_heads)
    BLOCK_DMODEL = triton.next_power_of_two(head_dim)
    
    # Calculate grid size for kernel launch
    num_blocks_head = triton.cdiv(num_heads, BLOCK_HEAD)
    num_blocks_dmodel = triton.cdiv(head_dim, BLOCK_DMODEL)
    grid = (seq_len * num_blocks_head * num_blocks_dmodel,)
    
    # Launch kernel with calculated parameters
    _fwd_kernel_destindex_copy_kv[grid](
        K, Out, DestLoc,
        K.stride(0), K.stride(1), K.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        seq_len, num_heads, head_dim,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_DMODEL=BLOCK_DMODEL
    )
