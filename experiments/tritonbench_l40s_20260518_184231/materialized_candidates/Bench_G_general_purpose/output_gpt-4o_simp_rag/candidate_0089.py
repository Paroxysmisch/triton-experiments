import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K, Dest_loc,
    Out,
    stride_k_bs, stride_k_h, stride_k_d,
    stride_o_bs, stride_o_h, stride_o_d,
    head_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr
):
    # Get the current program index (sequence index in the batch)
    cur_index = tl.program_id(0)
    
    # Define offsets for head and depth dimensions
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load the destination index for the current sequence
    dest_index = tl.load(Dest_loc + cur_index)

    # Calculate pointers for source and destination tensors
    k_ptrs = K + cur_index * stride_k_bs + stride_k_h * offs_h[:, None] + stride_k_d * offs_d[None, :]
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]

    # Load data from the source tensor K
    k = tl.load(k_ptrs, mask=offs_h[:, None] < head_num, other=0.0)
    
    # Store the loaded data into the destination tensor Out
    tl.store(o_ptrs, k, mask=offs_h[:, None] < head_num)
    return

@torch.no_grad()
def destindex_copy_kv(K, DestLoc, Out):
    # Get dimensions from the input tensors
    seq_len = DestLoc.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    
    # Ensure the dimensions of K and Out match
    assert K.shape[1] == Out.shape[1] and K.shape[2] == Out.shape[2]
    
    # Determine block size for heads
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    
    # Define the grid size for the kernel execution
    grid = (seq_len,)
    num_warps = 1

    # Launch the Triton kernel
    _fwd_kernel_destindex_copy_kv[grid](
        K, DestLoc, Out,
        K.stride(0), K.stride(1), K.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        head_num,
        BLOCK_DMODEL=head_dim,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )
    return
