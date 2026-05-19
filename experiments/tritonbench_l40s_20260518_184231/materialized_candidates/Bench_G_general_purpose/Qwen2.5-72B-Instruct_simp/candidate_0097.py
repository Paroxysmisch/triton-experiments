import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K,  # source tensor
    Out,  # destination tensor
    DestLoc,  # target indices in the batch dimension
    stride_k_batch,  # stride of K in the batch dimension
    stride_k_head,  # stride of K in the head dimension
    stride_k_depth,  # stride of K in the depth dimension
    stride_out_batch,  # stride of Out in the batch dimension
    stride_out_head,  # stride of Out in the head dimension
    stride_out_depth,  # stride of Out in the depth dimension
    stride_destloc_batch,  # stride of DestLoc in the batch dimension
    stride_destloc_head,  # stride of DestLoc in the head dimension
    batch_size,  # batch size
    head_size,  # number of heads
    depth_size,  # depth size
    BLOCK_HEAD: tl.constexpr,  # block size for head dimension
    BLOCK_DMODEL: tl.constexpr  # block size for depth dimension
):
    # Get the current block's head and depth indices
    head_pid = tl.program_id(axis=0)
    depth_pid = tl.program_id(axis=1)

    # Compute the block's range in the head and depth dimensions
    head_range = head_pid * BLOCK_HEAD, (head_pid + 1) * BLOCK_HEAD
    depth_range = depth_pid * BLOCK_DMODEL, (depth_pid + 1) * BLOCK_DMODEL

    # Iterate over the batch dimension
    for batch_idx in range(batch_size):
        dest_loc = tl.load(DestLoc + batch_idx * stride_destloc_batch + head_pid * stride_destloc_head)

        # Iterate over the head and depth dimensions within the block
        for head_idx in range(*head_range):
            for depth_idx in range(*depth_range):
                if head_idx < head_size and depth_idx < depth_size:
                    k_idx = batch_idx * stride_k_batch + head_idx * stride_k_head + depth_idx * stride_k_depth
                    out_idx = dest_loc * stride_out_batch + head_idx * stride_out_head + depth_idx * stride_out_depth
                    out_val = tl.load(K + k_idx)
                    tl.store(Out + out_idx, out_val)

import torch

def destindex_copy_kv(K, Out, DestLoc, BLOCK_HEAD=32, BLOCK_DMODEL=32):
    # Get the dimensions of the tensors
    batch_size, head_size, depth_size = K.shape

    # Define the grid and block dimensions
    grid = (head_size // BLOCK_HEAD, depth_size // BLOCK_DMODEL)

    # Call the Triton kernel
    _fwd_kernel_destindex_copy_kv[grid](
        K,  # source tensor
        Out,  # destination tensor
        DestLoc,  # target indices in the batch dimension
        K.stride(0),  # stride of K in the batch dimension
        K.stride(1),  # stride of K in the head dimension
        K.stride(2),  # stride of K in the depth dimension
        Out.stride(0),  # stride of Out in the batch dimension
        Out.stride(1),  # stride of Out in the head dimension
        Out.stride(2),  # stride of Out in the depth dimension
        DestLoc.stride(0),  # stride of DestLoc in the batch dimension
        DestLoc.stride(1),  # stride of DestLoc in the head dimension
        batch_size,  # batch size
        head_size,  # number of heads
        depth_size,  # depth size
        BLOCK_HEAD,  # block size for head dimension
        BLOCK_DMODEL  # block size for depth dimension
    )
