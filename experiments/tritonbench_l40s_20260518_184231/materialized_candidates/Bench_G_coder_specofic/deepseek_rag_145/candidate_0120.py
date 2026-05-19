import torch
import triton
import triton.language as tl

@triton.autotune()
@triton.jit
def chunk_global_cumsum_scalar_kernel(
    s,
    z,
    s_s_h,
    s_s_t,
    T: tl.constexpr,
    BT: tl.constexpr
):
    p_s = tl.program_id(0)  # grid stride
    # compute block and global indices
    i_b, i_h = tl.div(p_s, T), tl.mod(p_s, T)
    p_s = tl.make_block_ptr(s, (T,), (s_s_t,), (i_b * BT, i_h), (BT,))
    p_z = tl.make_block_ptr(z, (T,), (s_s_t,), (i_b * BT, i_h), (BT,))
    # load block
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    # compute cumulative sum
    b_z = tl.cumsum(b_s, 0)
    # store block
    tl.store(p_z, b_z.to(tl.float32))

def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    dtype: torch.dtype = None
) -> torch.Tensor:
    T, = s.shape
    dtype = dtype or s.dtype
    BT = 64  # block size
    z = torch.empty_like(s, dtype=dtype)
    grid = (T * s.shape[1],)
    chunk_global_cumsum_scalar_kernel[grid](s, z, s.stride(1), s.stride(2), T=T, BT=BT)
    return z
