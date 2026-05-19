import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def solve_multiple_lu_kernel(
    A,
    Bs,
    As,
    Bs,
    xs,
    p,
    n_cols,
    n_rows,
    n_cols_rounded,
    n_rows_log2,
    n_rows_rounded,
    pivot: tl.constexpr,
):
    batch_id = tl.program_id(0)
    row_block_id = tl.program_id(1)
    col_block_id = tl.program_id(2)
    n_rows_per_block = tl.num_programs(1)
    n_cols_per_block = tl.num_programs(2)

    # Compute the row and column offsets for the current block
    row_offsets = row_block_id * n_rows_per_block + tl.arange(0, n_rows_per_block)
    col_offsets = col_block_id * n_cols_per_block + tl.arange(0, n_cols_per_block)

    # Load the block of A and B that we are going to solve for
    A_block = tl.load(
        A
        + batch_id * As
        + row_offsets[:, None] * n_cols + col_offsets[None, :],
        mask=(row_offsets[:, None] < n_rows) & (col_offsets[None, :] < n_cols),
        other=0.0,
    )
    B_block = tl.load(
        B
        + batch_id * Bs
        + row_offsets[:, None] * n_cols,
        mask=row_offsets[:, None] < n_rows,
        other=0.0,
    )

    # If pivoting is enabled, load the pivot indices and reorder the A and B blocks
    if pivot:
        p_block = tl.load(
            p
            + batch_id * ps
            + row_offsets,
            mask=row_offsets < n_rows,
            other=0,
        )
        A_block = tl.trans(tl.trans(A_block, row_offsets, p_block))
        B_block = tl.trans(B_block, row_offsets, p_block)

    # Compute the LU decomposition of the block of A
    l_mask = tl.arange(0, n_rows_per_block)[:, None] >= tl.arange(0, n_cols_per_block)[None, :]
    u_mask = tl.logical_not(l_mask)
    A_block = tl.where(l_mask, A_block, 0.0)
    A_block = tl.where(u_mask, A_block, 0.0)

    # Solve the block of linear systems A x = B for the block of x
    for i in range(0, n_rows_per_block):
        for j in range(0, n_cols_per_block):
            b_block = tl.load(
                B_block
                + i * n_cols
                + j,
                mask=i < n_rows,
                other=0.0,
            )
            x_block = tl.sum(A_block[i, :] * b_block[:, None], 0)
            tl.store(
                x
                + batch_id * xs
                + i * n_cols
                + j,
                x_block,
                mask=i < n_rows,
            )


def solve_multiple_lu(
    A: Tensor,
    Bs: Tensor,
    *,
    pivot: bool = True,
    out: Optional[Tensor] = None,
) -> Tensor:
    # Check that A is a square matrix
    n_rows, n_cols = A.shape[-2:]
    assert n_rows == n_cols

    # Round up the number of rows and columns to the nearest power of 2
    n_rows_log2 = tl.cdiv(tl.cdiv(n_rows, 2), 2)
    n_rows_rounded = tl.cdiv(n_rows, 2**n_rows_log2)
    n_cols_rounded = tl.cdiv(n_cols, 2**n_rows_log2)

    # Calculate the number of programs needed to cover the block size
    num_programs = (n_rows_rounded, n_cols_rounded)

    # Create an output tensor if one was not provided
    if out is None:
        x = torch.empty_like(Bs)
    else:
        x = out

    # Solve the linear systems using a triton kernel
    grid = lambda meta: (Bs.shape[0],) + num_programs
    solve_multiple_lu_kernel[grid](
        A,
        Bs,
        *Bs.stride(),
        x,
        *x.stride(),
        *A.stride(),
        pivot=pivot,
        n_cols=n_cols,
        n_rows=n_rows,
        n_cols_rounded=n_cols_rounded,
        n_rows_log2=n_rows_log2,
        n_rows_rounded=n_rows_rounded,
    )

    return x
