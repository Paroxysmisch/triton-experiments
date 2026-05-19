import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'TILE_M': 16, 'TILE_N': 16, 'TILE_K': 32, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'TILE_M': 16, 'TILE_N': 16, 'TILE_K': 32, 'GROUP_M': 8}, num_warps=8),
        triton.Config({'TILE_M': 16, 'TILE_N': 16, 'TILE_K': 32, 'GROUP_M': 8}, num_warps=16),
        triton.Config({'TILE_M': 16, 'TILE_N': 16, 'TILE_K': 64, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'TILE_M': 16, 'TILE_N': 16, 'TILE_K': 64, 'GROUP_M': 8}, num_warps=8),
        triton.Config({'TILE_M': 16, 'TILE_N': 16, 'TILE_K': 64, 'GROUP_M': 8}, num_warps=16),
        triton.Config({'TILE_M': 32, 'TILE_N': 32, 'TILE_K': 32, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'TILE_M': 32, 'TILE_N': 32, 'TILE_K': 32, 'GROUP_M': 8}, num_warps=8),
        triton.Config({'TILE_M': 32, 'TILE_N': 32, 'TILE_K': 32, 'GROUP_M': 8}, num_warps=16),
        triton.Config({'TILE_M': 32, 'TILE_N': 32, 'TILE_K': 64, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'TILE_M': 32, 'TILE_N': 32, 'TILE_K': 64, 'GROUP_M': 8}, num_warps=8),
        triton.Config({'TILE_M': 32, 'TILE_N': 32, 'TILE_K': 64, 'GROUP_M': 8}, num_warps=16),
        triton.Config({'TILE_M': 64, 'TILE_N': 64, 'TILE_K': 64, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'TILE_M': 64, 'TILE_N': 64, 'TILE_K': 64, 'GROUP_M': 8}, num_warps=8),
        triton.Config({'TILE_M': 64, 'TILE_N': 64, 'TILE_K': 64, 'GROUP_M': 8}, num_warps=16),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bmm_kernel(
    A, B, O, M, N, K,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):
    TM = tl.program_id(0)
    TN = tl.program_id(1)
    TK = tl.program_id(2)

    # compute tile indices
    if not DIVISIBLE_M:
        TM %= GROUP_M
    if not DIVISIBLE_N:
        TN %= GROUP_M

    # range of indices for each tile
    tile_A_batch_dim = tl.arange(0, TILE_M)[:, None] * TILE_K * TILE_N
    tile_B_batch_dim = tl.arange(0, TILE_N)[None, :] * TILE_K * TILE_M
    tile_A_m_dim = tl.arange(0, TILE_M)[:, None] * TILE_K + tl.arange(0, TILE_K)[None, :]
    tile_A_n_dim = tl.arange(0, TILE_K)[:, None] * TILE_M + tl.arange(0, TILE_K)[None, :]
    tile_B_m_dim = tl.arange(0, TILE_K)[:, None] * TILE_N + tl.arange(0, TILE_K)[None, :]
    tile_B_n_dim = tl.arange(0, TILE_N)[None, :] * TILE_K + tl.arange(0, TILE_K)[None, :]

    # batch dimension indices
    batch_idx = TM * GROUP_M + tl.arange(0, TILE_M)[:, None]

    # initialize offsets for A and B
    offs_A = 0 + tile_A_batch_dim + tile_A_m_dim * K + tile_A_n_dim
    offs_B = 0 + tile_B_batch_dim + tile_B_m_dim * K + tile_B_n_dim

    # mask for partial tiles
    tile_mask_A = (batch_idx < M)[:, :, None]
    tile_mask_B = (batch_idx < N)[:, None, :]
    tile_mask = (tile_mask_A & tile_mask_B).to(tl.int1)

    # pointers to A and B
    p_A = A + offs_A
    p_B = B + offs_B

    # accumulate output
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    tile_K = K
    if not DIVISIBLE_K:
        tile_K = tl.minimum(K - TK * TILE_K, TILE_K)

    # loop over K
    for k in range(0, tile_K, TILE_K):
        mask_k = (k + tl.arange(0, TILE_K)) < K
        # prefetch A and B
        A = tl.load(p_A, mask=((batch_idx < M)[:, :, None]) & mask_k, other=0.0)
        B = tl.load(p_B, mask=mask_k[None, :] & ((batch_idx < N)[:, None, :]), other=0.0)
        # accumulate
        acc += tl.dot(A, B, allow_tf32=True)
        # update pointers
        p_A += TILE_K
        p_B += TILE_K

    if DIVISIBLE_K:
        O = acc.to(O.dtype.element_ty)
    else:
        O = acc.to(tl.float16)

    # set up output tiles
    tile_O_batch_dim = tl.arange(0, TILE_M)[:, None] * N + tl.arange(0, TILE_N)[None, :]
    tile_O_m_dim = tl.arange(0, TILE_M)[:, None]
    tile_O_n_dim = tl.arange(0, TILE_N)[None, :]

    offs_O = TM * GROUP_M * N + tile_O_batch_dim + tile_O_m_dim * N + tile_O_n_dim

    # mask for partial tiles
    tile_mask_O = ((batch_idx * N + tile_O_batch_dim) < (M * N))[:, :, None]

    # output pointer
    p_O = O + offs_O

    tl.store(p_O, O, mask=tile_mask_O & tile_mask)


def bmm(A, B):
    batch, M, K = A.shape
    _, _, N = B.shape
    O = torch.empty((batch, M, N), device=A.device, dtype=A.dtype)

    # autotune config
    config = {
        'num_stages': 1,
        'num_warps': 4,
    }

    # grid dimensions
    def grid(meta):
        return (
            triton.cdiv(M, meta['TILE_M']) * triton.cdiv(N, meta['TILE_N']),
            triton.cdiv(K, meta['TILE_K']),
            1,
        )

    # ensure A and B are contiguous in memory
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

    # launch kernel
    bmm_kernel[grid](
        A, B, O, M, N, K,
        GROUP_M=8,
        DIVISIBLE_M=(M % 32 == 0),
        DIVISIBLE_N=(N % 32 == 0),
        DIVISIBLE_K=(K % 32 == 0),
        **config,
    )

    return O
