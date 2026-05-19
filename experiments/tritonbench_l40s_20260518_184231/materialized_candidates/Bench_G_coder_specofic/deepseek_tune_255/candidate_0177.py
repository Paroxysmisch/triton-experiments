import torch
import triton
import triton.language as tl

@triton.jit
def chunk_global_cumsum_vector_kernel(
    s,
    z,
    stride,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    pos = tl.program_id(0)
    m_s = tl.tril(tl.ones((BT, BT), dtype=tl.float32))
    p_s = tl.make_block_ptr(s + pos * stride, (BS, BT), (BT, 1), (0, 0), (BS, BT), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_c = tl.dot(m_s, b_s)
    p_z = tl.make_block_ptr(z + pos * stride, (BS, BT), (BT, 1), (0, 0), (BS, BT), (1, 0))
    tl.store(p_z, b_c.to(p_z.dtype.element_ty))

def chunk_global_cumsum_vector(s: torch.Tensor):
    B, H, T, S = s.shape
    BT = 32
    BS = 64
    num_warps = 4
    assert s.stride(2) == 1
    s = s.contiguous()
    z = torch.empty_like(s, dtype=torch.float32)
    chunk_global_cumsum_vector_kernel[(B * H, )](
        s,
        z,
        s.stride(1),
        BT=BT,
        BS=BS,
        num_warps=num_warps,
        num_stages=1,
    )
    return z
