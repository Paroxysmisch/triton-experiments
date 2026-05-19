import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BT": 16}, num_warps=2),
        triton.Config({"BT": 16}, num_warps=4),
        triton.Config({"BT": 16}, num_warps=8),
        triton.Config({"BT": 32}, num_warps=2),
        triton.Config({"BT": 32}, num_warps=4),
        triton.Config({"BT": 32}, num_warps=8),
        triton.Config({"BT": 64}, num_warps=2),
        triton.Config({"BT": 64}, num_warps=4),
        triton.Config({"BT": 64}, num_warps=8),
        triton.Config({"BT": 128}, num_warps=2),
        triton.Config({"BT": 128}, num_warps=4),
        triton.Config({"BT": 128}, num_warps=8),
        triton.Config({"BT": 256}, num_warps=2),
        triton.Config({"BT": 256}, num_warps=4),
        triton.Config({"BT": 256}, num_warps=8)
    ],
    key=["ST", "BS"]
)
@triton.jit
def chunk_global_reversed_cumsum_vector_kernel(
    s,
    z,
    ST: tl.constexpr,
    BS: tl.constexpr,
    BT: tl.constexpr
):
    i_s, i_bh = tl.program_id(0), tl.program_id(1)
    b_s = tl.zeros([BS], dtype=tl.float32)
    m_s = tl.arange(0, BT) <= tl.cdiv(ST, BT) - 1

    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_s = tl.make_block_ptr(s + i_bh * (H * T * S) + i_t * BT * S + i_s * BS, (H, T, S),
                                (T * S, S, 1), (i_s * BS, i_t * BT, 0), (BS, BT, 1), (0, 1, 0))
        p_z = tl.make_block_ptr(z + i_bh * (H * T * S) + i_t * BT * S + i_s * BS, (H, T, S),
                                (T * S, S, 1), (i_s * BS, i_t * BT, 0), (BS, BT, 1), (0, 1, 0))
        if i_t == tl.cdiv(T, BT) - 1:
            b_s += tl.sum(tl.load(p_s, boundary_check=(0, 1, 2)) * m_s[None, None, :], axis=1)
        else:
            b_s += tl.load(p_s, boundary_check=(0, 1, 2)) * m_s[None, None, :]
        tl.store(p_z, b_s[None, :, None], boundary_check=(0, 1))
    return

@torch.no_grad()
def chunk_global_reversed_cumsum_vector(s, dtype=None):
    B, H, T, S = s.shape
    BS = 32
    if dtype is None:
        dtype = s.dtype
    z = torch.empty_like(s, dtype=dtype)
    NT = triton.cdiv(T, BS)
    def grid(meta): return (triton.cdiv(meta["BS"], meta["num_warps"]), B * H)
    chunk_global_reversed_cumsum_vector_kernel[grid](
        s, z,
        ST=T, BS=BS
    )
    return z
