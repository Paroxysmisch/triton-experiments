import triton
import triton.language as tl
import torch

@triton.jit
def _det_kernel(
    A_ptr,  # pointer to the first element of the current matrix in batch
    out_ptr,  # pointer to the output for the determinant of the current matrix
    n,       # dimension of the matrix (n x n)
    stride_row,  # distance in memory between consecutive rows of A
    stride_out,  # (unused) if out was batched
    batch_idx,   # index of the matrix in the batch
    BLOCK_SIZE: tl.constexpr
):
    # We assume one block processes a single matrix in the batch.
    # This kernel performs an in-place LU decomposition with partial pivoting
    # in a naive manner, then computes the product of diagonal elements.

    # Each thread will handle multiple elements in a row-based fashion.
    # We'll store pivot information in a local array pivot[] for row swaps.

    # row offset in A for this batch
    # We shift base pointer by batch_idx * n * stride_row
    A_base = A_ptr + batch_idx * n * stride_row

    # We'll use local memory for pivot indices
    pivot = tl.zeros([n], dtype=tl.int32)
    # Initialize pivot with 0..n-1
    for i in range(n):
        pivot[i] = i

    sign = 1.0  # track sign changes caused by row swaps

    # For each row, do partial pivoting, then factor
    for i in range(n):
        # Find pivot row
        max_val = tl.abs(tl.load(A_base + pivot[i]*stride_row + i, mask=True, other=0.0))
        pivot_row = i
        for r in range(i+1, n):
            val = tl.abs(tl.load(A_base + pivot[r]*stride_row + i, mask=True, other=0.0))
            if val > max_val:
                pivot_row = r
                max_val = val

        # If pivot_row != i, swap pivot indices and update sign
        if pivot_row != i:
            tmp = pivot[i]
            pivot[i] = pivot[pivot_row]
            pivot[pivot_row] = tmp
            sign = -sign

        # Eliminate below
        pivot_i = pivot[i]
        diag_val = tl.load(A_base + pivot_i*stride_row + i)
        # If the diagonal is zero, determinant is zero
        if diag_val == 0:
            # We can zero out the product and leave
            if tl.thread_idx().x == 0:
                tl.store(out_ptr + batch_idx, 0.0)
            return

        for r in range(i+1, n):
            pivot_r = pivot[r]
            row_val = tl.load(A_base + pivot_r*stride_row + i)
            factor = row_val / diag_val
            # store updated row
            for c in range(i, n, BLOCK_SIZE):
                c_idx = c + tl.thread_idx().x
                if c_idx < n:
                    val_rc = tl.load(A_base + pivot_r*stride_row + c_idx)
                    val_ic = tl.load(A_base + pivot_i*stride_row + c_idx)
                    new_rc = val_rc - factor * val_ic
                    tl.store(A_base + pivot_r*stride_row + c_idx, new_rc)

    # Compute product of diagonal
    det_val = sign
    for i in range(n):
        pivot_i = pivot[i]
        diag_val = tl.load(A_base + pivot_i*stride_row + i)
        det_val = det_val * diag_val

    # Store determinant
    if tl.thread_idx().x == 0:
        tl.store(out_ptr + batch_idx, det_val)


def det(A, *, out=None):
    """
    linalg.det(A, *, out=None) -> Tensor
    Computes the determinant of a square matrix A.
    A (Tensor): tensor of shape (*, n, n) where * is zero or more batch dimensions
    out (Tensor, optional): output tensor. Ignored if None. Default: None
    """
    # Ensure input is a tensor
    if not isinstance(A, torch.Tensor):
        raise TypeError("A must be a torch.Tensor")

    # Check shape
    if A.dim() < 2:
        raise RuntimeError("Input must be at least 2D")
    n = A.shape[-1]
    if A.shape[-2] != n:
        raise RuntimeError("Last two dimensions of A must form a square matrix")

    # Flatten batch dimensions
    batch_size = 1
    for s in A.shape[:-2]:
        batch_size *= s

    # Prepare output
    if out is None:
        out = torch.empty(A.shape[:-2], dtype=A.dtype, device=A.device)
    else:
        if out.shape != A.shape[:-2]:
            raise RuntimeError("out must have the same batch shape as A without the last two dims")

    # Convert to contiguous for Triton
    A_contig = A.contiguous()
    # We'll store the results in out (also contiguous if needed)
    out_contig = out.contiguous()

    # Launch kernel for each matrix in the batch
    # We let one block handle one matrix
    BLOCK_SIZE = 32  # arbitrary block size
    grid = lambda meta: (batch_size, )
    _det_kernel[grid](
        A_contig.data_ptr(),
        out_contig.data_ptr(),
        n,
        A_contig.stride(-2),  # row stride
        1,  # stride_out (unused in this demo)
        tl.arange(0, batch_size),  # each block reads its batch index
        BLOCK_SIZE = BLOCK_SIZE
    )

    # Reshape out_contig to the original batch shape
    if out is not out_contig:
        out.copy_(out_contig.view(*A.shape[:-2]))

    return out
