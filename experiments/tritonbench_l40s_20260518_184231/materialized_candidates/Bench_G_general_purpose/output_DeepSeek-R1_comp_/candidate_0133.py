import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K_ptr, Out_ptr, Dest_loc_ptr,
    K_batch_stride, K_head_stride, K_depth_stride,
    Out_seq_stride, Out_head_stride, Out_depth_stride,
    head_num: int, depth: int,
    BLOCK_HEAD: tl.constexpr,
):
    pid = tl.program_id(0)  # Sequential position in Dest_loc
    batch_idx = tl.load(Dest_loc_ptr + pid)  # Get batch index from Dest_loc
    
    # Compute base pointers for current sequence position
    k_base = K_ptr + batch_idx * K_batch_stride
    o_base = Out_ptr + pid * Out_seq_stride
    
    # Generate head offset ranges with masking
    h_offs = tl.arange(0, BLOCK_HEAD)
    mask = h_offs < head_num
    
    # Generate depth offsets (full range since depth is fixed)
    d_offs = tl.arange(0, depth)
    
    # Compute 2D pointer arrays [BLOCK_HEAD, depth]
    k_ptrs = k_base + (h_offs[:, None] * K_head_stride) + (d_offs[None, :] * K_depth_stride)
    o_ptrs = o_base + (h_offs[:, None] * Out_head_stride) + (d_offs[None, :] * Out_depth_stride)
    
    # Vectorized copy with head masking
    k_val = tl.load(k_ptrs, mask=mask[:, None], other=0.0)
    tl.store(o_ptrs, k_val, mask=mask[:, None])


def destindex_copy_kv(K: torch.Tensor, Out: torch.Tensor, Dest_loc: torch.Tensor):
    # Validate tensor dimensions
    assert K.shape[1:] == Out.shape[1:], "K and Out must match head_num and depth dimensions"
    assert Dest_loc.shape[0] == Out.shape[0], "Dest_loc length must match Out sequence length"
    
    seq_len, head_num, depth = Out.shape
    BLOCK_HEAD = triton.next_power_of_2(head_num)  # Optimal block size for heads
    
    # Configure kernel grid (1D grid of sequence positions)
    grid = (seq_len,)
    
    # Launch kernel with optimized parameters
    _fwd_kernel_destindex_copy_kv[grid](
        K, Out, Dest_loc,
        K.stride(0), K.stride(1), K.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        head_num, depth,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=1  # Single warp execution for optimal occupancy
    )
