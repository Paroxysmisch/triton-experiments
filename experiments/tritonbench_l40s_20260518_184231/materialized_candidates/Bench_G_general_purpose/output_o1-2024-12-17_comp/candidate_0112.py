import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_kv(
    KV_nope_ptr, KV_rope_ptr, DestLoc_ptr, O_nope_ptr, O_rope_ptr,
    B, N, D_nope, D_rope,
    strideKV_nope_b, strideKV_nope_n, strideKV_nope_d,
    strideKV_rope_b, strideKV_rope_n, strideKV_rope_d,
    strideDestLoc_b, strideDestLoc_n,
    strideO_nope_b, strideO_nope_n, strideO_nope_d,
    strideO_rope_b, strideO_rope_n, strideO_rope_d,
    BLOCK_DMODEL_NOPE, BLOCK_DMODEL_ROPE,
    **META
):
    pid = tl.program_id(0)
    b = pid // N
    n = pid % N

    dest = tl.load(DestLoc_ptr + b * strideDestLoc_b + n * strideDestLoc_n)

    idx_nope = tl.arange(0, BLOCK_DMODEL_NOPE)
    idx_rope = tl.arange(0, BLOCK_DMODEL_ROPE)

    # Load/store KV_nope -> O_nope
    baseKV_nope = KV_nope_ptr + b * strideKV_nope_b + n * strideKV_nope_n
    baseO_nope = O_nope_ptr + b * strideO_nope_b + dest * strideO_nope_n
    mask_nope = idx_nope < D_nope
    val_nope = tl.load(baseKV_nope + idx_nope * strideKV_nope_d, mask=mask_nope, other=0.0)
    tl.store(baseO_nope + idx_nope * strideO_nope_d, val_nope, mask=mask_nope)

    # Load/store KV_rope -> O_rope
    baseKV_rope = KV_rope_ptr + b * strideKV_rope_b + n * strideKV_rope_n
    baseO_rope = O_rope_ptr + b * strideO_rope_b + dest * strideO_rope_n
    mask_rope = idx_rope < D_rope
    val_rope = tl.load(baseKV_rope + idx_rope * strideKV_rope_d, mask=mask_rope, other=0.0)
    tl.store(baseO_rope + idx_rope * strideO_rope_d, val_rope, mask=mask_rope)


def destindex_copy_kv(KV_nope, KV_rope, DestLoc, O_nope, O_rope):
    # Ensure input and output shapes match
    B, N, D_nope = KV_nope.shape
    B2, N2, D_rope = KV_rope.shape
    B3, N3, D_nope2 = O_nope.shape
    B4, N4, D_rope2 = O_rope.shape

    assert B == B2 == B3 == B4, "Mismatched batch dimensions"
    assert N == N2 == N3 == N4, "Mismatched head dimensions"
    assert D_nope == D_nope2, "Mismatched KV_nope and O_nope hidden dimensions"
    assert D_rope == D_rope2, "Mismatched KV_rope and O_rope hidden dimensions"

    # Next power-of-two alignment for block sizes
    BLOCK_DMODEL_NOPE = 2 ** ((D_nope - 1).bit_length())
    BLOCK_DMODEL_ROPE = 2 ** ((D_rope - 1).bit_length())

    # Strides
    strideKV_nope_b, strideKV_nope_n, strideKV_nope_d = KV_nope.stride()
    strideKV_rope_b, strideKV_rope_n, strideKV_rope_d = KV_rope.stride()
    strideO_nope_b, strideO_nope_n, strideO_nope_d = O_nope.stride()
    strideO_rope_b, strideO_rope_n, strideO_rope_d = O_rope.stride()
    strideDestLoc_b, strideDestLoc_n = DestLoc.stride()  # Assuming DestLoc is 2D: (B, N)

    grid = (B * N,)

    _fwd_kernel_destindex_copy_kv[grid](
        KV_nope, KV_rope, DestLoc, O_nope, O_rope,
        B, N, D_nope, D_rope,
        strideKV_nope_b, strideKV_nope_n, strideKV_nope_d,
        strideKV_rope_b, strideKV_rope_n, strideKV_rope_d,
        strideDestLoc_b, strideDestLoc_n,
        strideO_nope_b, strideO_nope_n, strideO_nope_d,
        strideO_rope_b, strideO_rope_n, strideO_rope_d,
        BLOCK_DMODEL_NOPE, BLOCK_DMODEL_ROPE,
        num_warps=4,
        num_stages=2
    )
