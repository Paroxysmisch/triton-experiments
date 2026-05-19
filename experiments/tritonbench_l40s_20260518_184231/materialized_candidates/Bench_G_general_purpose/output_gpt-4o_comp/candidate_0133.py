import triton
import triton.language as tl
import math

# Triton kernel for indexed copying
@triton.jit
def _fwd_kernel_destindex_copy_kv(
    dest_loc_ptr,  # Pointer to the destination index tensor (Dest_loc)
    k_ptr,         # Pointer to the source tensor (K)
    out_ptr,       # Pointer to the destination tensor (Out)
    stride_bs_k,   # Stride for batch size in K
    stride_h_k,    # Stride for head in K
    stride_d_k,    # Stride for depth in K
    stride_bs_o,   # Stride for batch size in Out
    stride_h_o,    # Stride for head in Out
    stride_d_o,    # Stride for depth in Out
    head_num,      # Number of heads
    depth,         # Depth dimension
    BLOCK_HEAD: tl.constexpr  # Block size for head dimension
):
    # Program ID corresponds to the sequence length index
    seq_idx = tl.program_id(0)

    # Compute offsets for Dest_loc
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, depth)

    # Load the destination head index from Dest_loc
    dest_loc = tl.load(dest_loc_ptr + seq_idx)

    # Mask for valid head indices
    mask_h = offs_h < head_num

    # Compute pointers for source (K) and destination (Out)
    k_ptrs = k_ptr + (dest_loc * stride_h_k + offs_h * stride_h_k)[:, None] + offs_d[None, :]
    o_ptrs = out_ptr + (seq_idx * stride_bs_o + offs_h * stride_h_o)[:, None] + offs_d[None, :]

    # Perform masked load from K and store into Out
    k_vals = tl.load(k_ptrs, mask=mask_h[:, None])
    tl.store(o_ptrs, k_vals, mask=mask_h[:, None])


# Wrapper function
def destindex_copy_kv(dest_loc, K, Out):
    """
    Wrapper function to invoke the Triton kernel for indexed copying.

    Parameters:
    - dest_loc: Tensor of destination indices (shape: [seq_len])
    - K: Source tensor (shape: [batch_size, head_num, depth])
    - Out: Destination tensor (shape: [seq_len, head_num, depth])
    """
    # Extract tensor dimensions
    batch_size, head_num, depth = K.shape
    seq_len, _, _ = Out.shape

    # Validate tensor shapes
    assert dest_loc.shape[0] == seq_len, "Dest_loc must have the same sequence length as Out."
    assert Out.shape[1] == head_num and Out.shape[2] == depth, "Out shape must match head_num and depth of K."

    # Compute strides for K and Out
    stride_bs_k = K.stride(0)
    stride_h_k = K.stride(1)
    stride_d_k = K.stride(2)

    stride_bs_o = Out.stride(0)
    stride_h_o = Out.stride(1)
    stride_d_o = Out.stride(2)

    # Compute BLOCK_HEAD as the next power of 2 of head_num for performance
    BLOCK_HEAD = 2 ** math.ceil(math.log2(head_num))

    # Configure grid size (1D grid along sequence length)
    grid = (seq_len,)

    # Launch the Triton kernel
    _fwd_kernel_destindex_copy_kv[grid](
        dest_loc_ptr=dest_loc,
        k_ptr=K,
        out_ptr=Out,
        stride_bs_k=stride_bs_k,
        stride_h_k=stride_h_k,
        stride_d_k=stride_d_k,
        stride_bs_o=stride_bs_o,
        stride_h_o=stride_h_o,
        stride_d_o=stride_d_o,
        head_num=head_num,
        depth=depth,
        BLOCK_HEAD=BLOCK_HEAD
    )
