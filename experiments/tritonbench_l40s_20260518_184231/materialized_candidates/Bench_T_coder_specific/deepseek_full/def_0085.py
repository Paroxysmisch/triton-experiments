import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.linalg import eigh

@triton.jit
def matrix_power_eig_kernel(A, k, O, n_blocks_x, n_blocks_y, M, N,
                             BLOCK_SIZE: tl.constexpr):
    # map the program id to the row of A and column of B it should compute
    row, col = tl.program_id(0), tl.program_id(1)
    r_block = row % n_blocks_x
    c_block = col % n_blocks_y
    row_block = row // n_blocks_x
    col_block = col // n_blocks_y

    offset_row = row_block * BLOCK_SIZE
    offset_col = col_block * BLOCK_SIZE

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for i in range(0, BLOCK_SIZE, 16):
        for j in range(0, BLOCK_SIZE, 16):
            k_vals = tl.load(A + (offset_row + i)[:, None] * M +
                             (offset_col + j)[None, :] +
                             tl.arange(0, 16)[:, None] * M +
                             tl.arange(0, 16)[None, :] * N)
            k_vals *= k
            k_vec = tl.sum(k_vals, axis=1)
            acc += tl.dot(k_vals, k_vec)

    acc = tl.where(row < N, acc, 0)
    acc = tl.where(col < N, acc, 0)

    tl.store(O + row * N + col, acc)


def matrix_power_eig(A: Tensor, k: float, *, out: Tensor = None) -> Tensor:
    A = A.unsqueeze(0) if A.ndim == 2 else A
    n_matrices, _, n = A.shape
    k = complex(k)
    A = A.view(n_matrices, -1).contiguous()

    vals, vecs = eigh(A)

    N = vals.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 4
    num_stages = 4 if BLOCK_SIZE <= 64 else 3
    num_ctas = n_matrices * N

    vals = vals.view(n_matrices, N, N).contiguous()

    vals_abs_max = (vals.abs().max(dim=-1)[0].max(dim=-1)[0]).unsqueeze(-1).unsqueeze(-1)
    vals_normalized = vals / vals_abs_max
    vals_normalized = vals_normalized.view(n_matrices, -1).contiguous()

    if out is None:
        out = torch.empty((n_matrices, N, N), dtype=A.dtype, device=A.device)
    else:
        out = out.view(n_matrices, -1).contiguous()

    grid = (n_matrices * N, n_matrices * N)

    matrix_power_eig_kernel[grid](vals_normalized, k, out, n_matrices, N // BLOCK_SIZE,
                                  N, N,
                                  BLOCK_SIZE=BLOCK_SIZE,
                                  num_warps=num_warps,
                                  num_stages=num_stages,
                                  num_ctas=num_ctas)

    out = out.view(n_matrices, N, N)

    return out
