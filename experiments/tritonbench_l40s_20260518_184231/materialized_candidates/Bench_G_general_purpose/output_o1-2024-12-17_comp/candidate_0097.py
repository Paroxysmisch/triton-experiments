import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K_ptr,               # *f32
    DestLoc_ptr,         # *int32
    Out_ptr,             # *f32
    B, H, D,             # dimensions: batch, heads, d_model
    stride_kb, stride_kh, stride_kd,   # strides for K
    stride_ob, stride_oh, stride_od,   # strides for Out
    BLOCK_HEAD: tl.constexpr,          # block size along heads dim
    BLOCK_DMODEL: tl.constexpr         # block size along d_model dim
):
    # Current batch index
    b_idx = tl.program_id(0)
    # Retrieve the destination batch index from DestLoc
    dest_b_idx = tl.load(DestLoc_ptr + b_idx)
    
    # Loop over heads
    # Each iteration covers BLOCK_HEAD heads at a time
    h_offset = 0
    while h_offset < H:
        # Loop over d_model
        # Each iteration covers BLOCK_DMODEL elements at a time
        d_offset = 0
        while d_offset < D:
            # Offsets within the current block
            off_h = tl.arange(0, BLOCK_HEAD)
            off_d = tl.arange(0, BLOCK_DMODEL)
            h_pos = h_offset + off_h
            d_pos = d_offset + off_d
            
            # Masks to guard against out-of-bounds
            mask_h = h_pos < H
            mask_d = d_pos < D
            mask = mask_h[:, None] & mask_d[None, :]

            # Compute pointer offsets for K and Out
            k_ptrs = K_ptr + (b_idx * stride_kb + h_pos * stride_kh)[:, None] + d_pos[None, :] * stride_kd
            o_ptrs = Out_ptr + (dest_b_idx * stride_ob + h_pos * stride_oh)[:, None] + d_pos[None, :] * stride_od

            # Load from K and store to Out
            k_vals = tl.load(k_ptrs, mask=mask, other=0.0)
            tl.store(o_ptrs, k_vals, mask=mask)

            d_offset += BLOCK_DMODEL
        h_offset += BLOCK_HEAD

def destindex_copy_kv(K: torch.Tensor, DestLoc: torch.Tensor, Out: torch.Tensor):
    B, H, D = K.shape
    assert DestLoc.shape[0] == B
    assert Out.shape == (B, H, D)

    # Example block sizes (powers of two for performance)
    BLOCK_HEAD = 32
    BLOCK_DMODEL = 64

    # Launch grid: one program per batch element
    grid = (B,)

    _fwd_kernel_destindex_copy_kv[grid](
        K, 
        DestLoc, 
        Out,
        B, H, D,
        K.stride(0), K.stride(1), K.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        BLOCK_HEAD=BLOCK_HEAD, 
        BLOCK_DMODEL=BLOCK_DMODEL
    )
