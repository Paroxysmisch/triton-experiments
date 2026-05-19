import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K_ptr, DestLoc_ptr, Out_ptr, Out_scale_ptr,
    seq_len, head_dim,
    K_stride_0, K_stride_1,
    DestLoc_stride_0,
    Out_stride_0, Out_stride_1,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the program ID
    pid = tl.program_id(0)

    # Compute the offsets for K and Out
    k_offset = pid * K_stride_0
    out_offset = tl.load(DestLoc_ptr + pid * DestLoc_stride_0) * Out_stride_0

    # Load a block of K
    k = tl.load(K_ptr + k_offset + tl.arange(0, BLOCK_SIZE) * K_stride_1)

    # Compute the scale (max absolute value / 127 to utilize int8 range)
    max_abs = tl.max(tl.abs(k))
    scale = max_abs / 127.0

    # Store the scale
    tl.store(Out_scale_ptr + pid, scale)

    # Quantize and store the output
    out = tl.cast(k / scale, tl.int8)
    tl.store(Out_ptr + out_offset + tl.arange(0, BLOCK_SIZE) * Out_stride_1, out)

def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale):
    seq_len, num_heads, head_dim = K.shape
    assert K.is_contiguous()
    assert DestLoc.is_contiguous()
    assert Out.is_contiguous()
    assert Out_scale.is_contiguous()
    assert K.dtype == torch.float32
    assert DestLoc.dtype == torch.int64
    assert Out.dtype == torch.int8
    assert Out_scale.dtype == torch.float32

    grid = (seq_len,)
    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K, DestLoc, Out, Out_scale,
        seq_len, head_dim,
        K.stride(0), K.stride(1),
        DestLoc.stride(0),
        Out.stride(0), Out.stride(1),
        BLOCK_SIZE=head_dim
    )

    return Out, Out_scale
