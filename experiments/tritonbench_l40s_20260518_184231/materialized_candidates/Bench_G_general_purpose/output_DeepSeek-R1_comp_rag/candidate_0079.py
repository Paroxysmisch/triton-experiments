import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    # Tensor pointers
    K,  # Input key/value tensor
    Dest_loc,  # Destination location indices
    Out,  # Output tensor
    
    # Memory strides for K
    stride_k_bs, stride_k_h, stride_k_d,
    # Memory strides for Out
    stride_o_bs, stride_o_h, stride_o_d,
    
    # Configuration parameters
    head_num: int,  # Number of attention heads
    BLOCK_DMODEL: tl.constexpr,  # Block size for head dimension (power of two)
    BLOCK_HEAD: tl.constexpr  # Block size for heads (power of two)
):
    # Each program processes one sequence position
    cur_index = tl.program_id(0)
    
    # Create block offsets for heads and dimensions
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Get destination index for current sequence position
    dest_index = tl.load(Dest_loc + cur_index)

    # Calculate pointer offsets for input and output
    k_ptr = (
        K + cur_index * stride_k_bs          # Batch/seq offset
        + offs_h[:, None] * stride_k_h        # Head offset
        + offs_d[None, :] * stride_k_d        # Dimension offset
    )
    
    o_ptr = (
        Out + dest_index * stride_o_bs        # Batch/seq offset
        + offs_h[:, None] * stride_o_h        # Head offset
        + offs_d[None, :] * stride_o_d        # Dimension offset
    )

    # Load and store with mask for valid heads
    k = tl.load(k_ptr, mask=offs_h[:, None] < head_num, other=0.0)
    tl.store(o_ptr, k, mask=offs_h[:, None] < head_num)

@torch.no_grad()
def destindex_copy_kv(K: torch.Tensor, DestLoc: torch.Tensor, Out: torch.Tensor):
    # Validate input dimensions
    assert K.dim() == 3, "Input K must be 3D (batch, head, dim)"
    assert Out.dim() == 3, "Output Out must be 3D (batch, head, dim)"
    assert K.size(1) == Out.size(1), "Head dimension mismatch"
    assert K.size(2) == Out.size(2), "Feature dimension mismatch"

    seq_len = DestLoc.size(0)
    head_num = K.size(1)
    head_dim = K.size(2)
    
    # Determine block sizes as next power of two for efficiency
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    
    # Configure kernel grid and launch parameters
    grid = (seq_len,)
    num_warps = 1  # Simple memory operations don't require many warps

    # Launch kernel with appropriate parameters
    _fwd_kernel_destindex_copy_kv[grid](
        K, DestLoc, Out,
        # Input strides
        K.stride(0), K.stride(1), K.stride(2),
        # Output strides
        Out.stride(0), Out.stride(1), Out.stride(2),
        # Configuration parameters
        head_num=K.size(1),
        BLOCK_DMODEL=head_dim,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=2  # Enough for moderate latency hiding
    )
    return Out
