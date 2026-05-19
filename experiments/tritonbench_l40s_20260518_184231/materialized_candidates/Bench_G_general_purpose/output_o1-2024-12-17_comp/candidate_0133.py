import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    k_ptr, 
    o_ptr, 
    dest_loc_ptr,
    stride_k_bs, 
    stride_k_sh, 
    stride_k_hd,
    stride_o_bs, 
    stride_o_sh, 
    stride_o_hd,
    B, 
    S, 
    H, 
    D, 
    BLOCK_HEAD,
    **meta
):
    pid0 = tl.program_id(0)
    dest_idx = tl.load(dest_loc_ptr + pid0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    head_mask = offs_h < H

    # Loop over batch and depth dimensions
    for b_ in range(B):
        for d_ in range(D):
            k_ptrs = k_ptr + b_ * stride_k_bs + pid0 * stride_k_sh + offs_h * stride_k_hd + d_
            o_ptrs = o_ptr + b_ * stride_o_bs + dest_idx * stride_o_sh + offs_h * stride_o_hd + d_
            val = tl.load(k_ptrs, mask=head_mask)
            tl.store(o_ptrs, val, mask=head_mask)

def destindex_copy_kv(K: torch.Tensor, Out: torch.Tensor, Dest_loc: torch.Tensor):
    B, S, H, D = K.shape
    assert Out.shape == (B, S, H, D), "Out must match K's shape."
    assert Dest_loc.shape[0] == S, "Dest_loc must have size seq_len."
    
    # Compute next power of 2 for head dimension
    def next_power_of_2(x):
        return 1 << (x - 1).bit_length()
    BLOCK_HEAD = next_power_of_2(H)

    grid = (S,)
    _fwd_kernel_destindex_copy_kv[grid](
        K, 
        Out, 
        Dest_loc,
        K.stride(0), 
        K.stride(1), 
        K.stride(2),
        Out.stride(0), 
        Out.stride(1), 
        Out.stride(2),
        B, 
        S, 
        H, 
        D, 
        BLOCK_HEAD,
        num_warps=1,
        num_stages=1
    )
