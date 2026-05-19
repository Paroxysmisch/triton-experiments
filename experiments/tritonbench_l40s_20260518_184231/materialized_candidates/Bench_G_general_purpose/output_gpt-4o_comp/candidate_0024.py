import triton
import triton.language as tl
import torch


@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(
    s_ptr, o_ptr,  # Pointers to input and output tensors
    T,             # Sequence length (dimension T)
    BT,            # Block size for T dimension
    BLOCK_M,       # Block size for batch/head dimension
    stride_sB, stride_sH, stride_sT,  # Strides for input tensor
    stride_oB, stride_oH, stride_oT,  # Strides for output tensor
    **meta
):
    # Get the block index in the grid
    b_h_idx = tl.program_id(0)  # (B * H) index
    # Calculate the batch and head indices
    b_idx = b_h_idx // meta['H']
    h_idx = b_h_idx % meta['H']

    # Compute offsets for the input/output pointers
    s_offset = b_idx * stride_sB + h_idx * stride_sH
    o_offset = b_idx * stride_oB + h_idx * stride_oH

    # Initialize accumulation variable for reversed cumulative sum
    b_z = 0.0

    # Iterate over the T dimension in reverse, in chunks of size BT
    for t_start in range(T, 0, -BT):
        # Determine the size of the current block (may be smaller at the beginning)
        t_end = max(0, t_start - BT)
        block_size = t_start - t_end

        # Load the current block of input values
        t_idxs = tl.arange(0, block_size)  # Local indices within the block
        s_block_ptr = s_ptr + s_offset + (t_end + t_idxs) * stride_sT
        s_block = tl.load(s_block_ptr, mask=t_idxs < block_size, other=0.0)

        # Compute the cumulative sum for the block
        b_z += tl.sum(s_block)
        o_block = b_z - tl.cumsum(s_block, axis=0)

        # Store the result in the output tensor
        o_block_ptr = o_ptr + o_offset + (t_end + t_idxs) * stride_oT
        tl.store(o_block_ptr, o_block, mask=t_idxs < block_size)


def chunk_global_reversed_cumsum_scalar(s, BT=128):
    """
    Wrapper function for the Triton kernel to compute the reversed cumulative sum.

    Args:
        s (torch.Tensor): Input tensor of shape (B, H, T).
        BT (int): Block size for the T dimension.

    Returns:
        torch.Tensor: Output tensor of the same shape as the input.
    """
    # Input tensor shape
    B, H, T = s.shape

    # Ensure the input is contiguous in memory
    s = s.contiguous()

    # Allocate output tensor
    o = torch.empty_like(s)

    # Define strides for input and output tensors
    stride_sB, stride_sH, stride_sT = s.stride()
    stride_oB, stride_oH, stride_oT = o.stride()

    # Launch the Triton kernel
    grid = (B * H,)  # Grid size (one block per (B, H) pair)
    chunk_global_reversed_cumsum_scalar_kernel[grid](
        s_ptr=s,
        o_ptr=o,
        T=T,
        BT=BT,
        BLOCK_M=1,  # Not used in this implementation, but can be extended
        stride_sB=stride_sB,
        stride_sH=stride_sH,
        stride_sT=stride_sT,
        stride_oB=stride_oB,
        stride_oH=stride_oH,
        stride_oT=stride_oT,
        H=H  # Pass H as a meta parameter
    )

    return o
