import triton
import triton.language as tl
import torch

@triton.jit
def _forward_solve_step_kernel(
    L_ptr,          # [*batch, n, n] flattened pointer to L
    B_ptr,          # [*batch, n, k] flattened pointer to B (solution in-place)
    i,              # current row index (solve step)
    n,              # number of rows in L
    k,              # number of columns in B
    strideLrow,     # stride for row in L
    strideLcol,     # stride for column in L
    strideBrow,     # stride for row in B
    strideBcol,     # stride for column in B
    BLOCK_SIZE: tl.constexpr
):
    """
    For a single row 'i', performs:
        B[i, :] = (B[i, :] - Σ_{j=0..i-1} L[i, j]*B[j, :]) / L[i, i]
    This kernel handles the summation for columns in parallel.
    """
    # program_id uniquely identifies the block along the columns dimension
    pid = tl.program_id(0)
    # range of columns this block will handle
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < k

    # Compute base pointers:
    # L[i, j] location for row i, we will loop j in [0..i-1] on the host side
    Lii_ptr = L_ptr + (i * strideLrow + i * strideLcol)
    # B[i, col]
    Bi_ptr = B_ptr + (i * strideBrow)
    # load L[i, i]
    Lii = tl.load(Lii_ptr)  # scalar

    # partial sum
    # sum_{j=0..i-1} L[i, j] * B[j, col]
    accum = tl.zeros([BLOCK_SIZE], dtype=tl.float32 if Lii.dtype == tl.float32 else Lii.dtype)
    # We'll do j in host loop calls for the partial sum,
    # so this kernel expects the host to call it after computing partial externally.
    # Here we just apply the partial correction. Example usage is in the Python loop.

    # B[i, col]
    Bi_val = tl.load(Bi_ptr + offs * strideBcol, mask=mask)
    # subtract the accum (provided in a separate pointer or so) if we had that,
    # but to keep it purely in device, we do a dedicated kernel for each j in Python.
    # For demonstration, we'll do it differently: pass accum via another pointer or register.
    # We'll store in final step. This snippet just shows final division to finalize row i solve.

    # This kernel is meant to just do B[i, col] = B[i, col] / L[i, i], so accum must be
    # pre-subtracted on B prior to this call in the host code. 
    # For brevity, we proceed to the final step:

    Bi_res = Bi_val / Lii
    tl.store(Bi_ptr + offs * strideBcol, Bi_res, mask=mask)


@triton.jit
def _forward_solve_accum_kernel(
    L_ptr,          
    B_ptr,          
    i,               # current target row
    j,               # row to accumulate from
    n,
    k,
    strideLrow,
    strideLcol,
    strideBrow,
    strideBcol,
    BLOCK_SIZE: tl.constexpr
):
    """
    For each column col in B:
        B[i, col] -=
