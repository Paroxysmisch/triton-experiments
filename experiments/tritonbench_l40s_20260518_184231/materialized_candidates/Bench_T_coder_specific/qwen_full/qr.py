import torch
import triton
import triton.language as tl

@triton.jit
def _qr_kernel_R(R, m, n, k, tiles_per_mat: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Compute indices
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Initialize pointers to matrices
    r = R + i[:, None] * n + k

    # Compute R
    for _ in range(tiles_per_mat):
        # Compute r
        r_norm = tl.math.sqrt(tl.sum(tl.math.conj(r) * r, 0))
        r = tl.where(i[:, None] >= n, r, r / r_norm)
        r = tl.where(i[:, None] < n, r, 0)

        # Update r
        r = tl.store(r, r)

@triton.jit
def _qr_kernel_QR(A, Q, R, m, n, k, tiles_per_mat_a: tl.constexpr, tiles_per_mat_q: tl.constexpr, tiles_per_mat_r: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Compute indices
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Initialize pointers to matrices
    a = A + i[:, None] * n + k
    qr = Q * R
    q = qr + i[:, None] * n + k
    r = R + i[:, None] * n + k

    # Compute QR
    for _ in range(tiles_per_mat_q):
        # Compute q
        a = tl.load(a)
        q = tl.dot(qr, a, allow_tf32=False)
        q = tl.store(q, q)

    for _ in range(tiles_per_mat_r):
        # Compute r
        a = tl.load(a)
        r = tl.dot(tl.math.conj(a), qr, allow_tf32=False)
        r = tl.store(r, r)

    for _ in range(tiles_per_mat_a):
        # Store a
        a = tl.load(a)
        tl.store(a, a)

def _qr(A: torch.Tensor, mode: str, out: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor]:
    m = A.shape[-2]
    n = A.shape[-1]
    k = min(m, n)
    num_batch_dims = A.ndim - 2
    num_tiles = k

    # Create views of the input tensor for tiling
    tiles_a = A.reshape(A.shape[:-2] + (num_tiles, -1))
    tiles_a = tiles_a.reshape(tiles_a.shape[:-1] + (triton.next_power_of_2(n),))
    tiles_a = tiles_a.contiguous()

    if mode == 'reduced':
        Q = torch.empty_like(tiles_a, dtype=A.dtype, device=A.device)
        R = torch.empty(tiles_a.shape[:-2] + (n, n), dtype=A.dtype, device=A.device)

        for i in range(num_tiles):
            tile_a = tiles_a[..., i, :]
            tile_q = Q[..., i, :]
            tile_r = R[..., i, :]

            tiles_per_mat_a = 1
            tiles_per_mat_q = 1
            tiles_per_mat_r = 1

            _qr_kernel_QR[(triton.cdiv(k, tiles_per_mat_a * triton.next_power_of_2(n)),)](tile_a, tile_q, tile_r, m, n, i, tiles_per_mat_a, tiles_per_mat_q, tiles_per_mat_r, 2 * triton.next_power_of_2(n))

        Q = torch.sum(Q, dim=-3)
        R = torch.sum(R, dim=-3)

    elif mode == 'complete':
        Q = torch.empty(tiles_a.shape, dtype=A.dtype, device=A.device)
        R = torch.empty(tiles_a.shape[:-2] + (n, n), dtype=A.dtype, device=A.device)

        for i in range(num_tiles):
            tile_a = tiles_a[..., i, :]
            tile_q = Q[..., i, :]
            tile_r = R[..., i, :]

            tiles_per_mat_a = 1
            tiles_per_mat_q = triton.cdiv(m, tiles_per_mat_a * n)
            tiles_per_mat_r = 1

            _qr_kernel_QR[(tiles_per_mat_q,)](tile_a, tile_q, tile_r, m, n, i, tiles_per_mat_a, tiles_per_mat_q, tiles_per_mat_r, 2 * triton.next_power_of_2(n))

        Q = torch.sum(Q, dim=-3)
        R = torch.sum(R, dim=-3)

    elif mode == 'r':
        Q = torch.empty([1] * num_batch_dims, dtype=A.dtype, device=A.device)
        R = torch.empty(tiles_a.shape, dtype=A.dtype, device=A.device)

        for i in range(num_tiles):
            tile_a = tiles_a[..., i, :]
            tile_r = R[..., i, :]

            tiles_per_mat_a = 1
            tiles_per_mat_r = 1

            _qr_kernel_R[(tiles_per_mat_a,)](tile_r, m, n, i, tiles_per_mat_a, 2 * triton.next_power_of_2(n))

        R = torch.sum(R, dim=-3)

    if out is not None:
        out[0].copy_(Q)
        out[1].copy_(R)
        return out

    else:
        return Q, R
