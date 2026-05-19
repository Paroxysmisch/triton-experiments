import torch
import triton
import triton.language as tl


@triton.jit
def det_fn(a_ptr, b_ptr, c_ptr, r_ptr, n, lda, ldb, ldc, incX, incY, incZ, inc_out, n_tiles, one_tile_per_cta: tl.constexpr,
           unsigned_int_dtype: tl.constexpr, **meta):
    _2N = 2 * n
    # These cycles unroll over the tiles.
    for tile_idx in range(0, n_tiles, inc_out if one_tile_per_cta else 1):
        # Compute the index of the diagonal element of the submatrix owned by this CTA.
        diag_idx = tile_idx * _2N + tl.arange(0, 1)
        # Load the submatrix owned by this CTA.
        a = tl.load(a_ptr + diag_idx * lda + tl.arange(0, n) * lda + tl.arange(0, n)[None, :], eviction_policy='evict_last',
                    other=unsigned_int_dtype(0)).to(tl.float32)
        b = tl.load(b_ptr + diag_idx * ldb + tl.arange(0, n) * ldb + tl.arange(0, n)[None, :], eviction_policy='evict_first',
                    other=unsigned_int_dtype(0)).to(tl.float32)
        c = tl.load(c_ptr + diag_idx * ldc + tl.arange(0, n) * ldc + tl.arange(0, n)[None, :], eviction_policy='evict_first',
                    other=unsigned_int_dtype(0)).to(tl.float32)
        # Compute the determinant of the submatrix.
        d = tl.dot(a, b, allow_tf32=False) * tl.dot(b, c, allow_tf32=False)
        # Write out the result.
        tl.store(r_ptr + diag_idx * incX + tl.arange(0, 1) * incX + tl.arange(0, n) * incY,
                 d.to(r_ptr.dtype.element_ty))


def det(A, *, out=None):
    f_name = "det"
    check_bsr_layout(f_name, A)
    if A.is_floating_point():
        signed_int_dtype = torch.promote_types(torch.int, A.dtype)
        unsigned_int_dtype = promote_to_unsigned(signed_int_dtype)
    else:
        A = A.to(torch.float)
        signed_int_dtype = torch.promote_types(torch.int, A.dtype)
        unsigned_int_dtype = promote_to_unsigned(signed_int_dtype)
    device = A.device
    # Prepare outputs.
    if out is not None:
        check_device(f_name, out, device)
        check_is_contiguous(f_name, out, True)
        check_dtype(f_name, out, A.dtype)
        check_shape(f_name, out, A.shape[:-2])
    else:
        out = torch.empty(A.shape[:-2], dtype=A.dtype, device=device)
    # Early return for empty tensors.
    if out.numel() == 0:
        return out
    # Get meta information about the algorithm.
    n = A.size(-1)
    meta = get_meta(A)
    # Launch kernel.
    def grid(meta): return (div_rn(meta['n_tiles'], meta['one_tile_per_cta']), 1, 1)
    inv_perm_tiled[grid](A, out, n, **meta)
    return out
