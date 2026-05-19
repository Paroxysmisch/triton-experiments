import torch
import triton
import triton.language as tl
from collections import namedtuple

MaxReturn = namedtuple("max_return", ["values", "indices"])

@triton.jit
def _max_argmax_kernel(
    input_ptr,  # *F32 or *F16
    output_vals_ptr,  # *F32 or *F16
    output_idx_ptr,   # *I64
    row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    # Pointers for this row
    row_input_ptr = input_ptr + row_idx * row_stride
    # We'll keep track of local maxima and indices
    # Initialize with the first valid element in the row
    init_val = tl.load(row_input_ptr + 0, mask=0 < n_cols, other=-float('inf'))
    max_val = init_val
    max_idx = 0

    # Loop over elements in BLOCK_SIZE chunks
    # Each iteration, we load one element for this program_id
    # and update the max + argmax if needed
    for offset in range(n_cols):
        val = tl.load(row_input_ptr + offset, mask=offset < n_cols, other=-float('inf'))
        cond = val > max_val
        max_val = tl.where(cond, val, max_val)
        max_idx = tl.where(cond, offset, max_idx)

    # Store results
    tl.store(output_vals_ptr + row_idx, max_val)
    tl.store(output_idx_ptr + row_idx, max_idx)

def max(input: torch.Tensor, dim: int, keepdim: bool = False, *, out=None):
    # Normalize dim
    ndim = input.dim()
    if dim < 0:
        dim += ndim

    # Permute input so that reduced dim is last
    # We'll just move `dim` to the end, flatten everything else as the leading dimension
    perm = list(range(ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x = input.permute(perm)
    shape = x.shape
    # Flatten
    rows = 1
    for s in shape[:-1]:
        rows *= s
    cols = shape[-1]

    # Prepare output for values/indices
    out_vals = torch.empty((rows,), dtype=x.dtype, device=x.device)
    out_idxs = torch.empty((rows,), dtype=torch.long, device=x.device)

    BLOCK_SIZE = 1  # We'll just loop internally in the kernel

    _max_argmax_kernel[(rows,)](
        x,
        out_vals,
        out_idxs,
        x.stride(0),
        cols,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape to the permuted shape with the last dim removed (since we reduced)
    out_vals = out_vals.view(*shape[:-1])
    out_idxs = out_idxs.view(*shape[:-1])

    # Undo the permutation
    # Insert the dimension back as 1 if keepdim is True; otherwise it's reduced
    inv_perm = list(range(ndim))
    inv_perm[-1], inv_perm[dim] = inv_perm[dim], inv_perm[-1]

    if keepdim:
        # Expand the last dim as 1
        out_vals = out_vals.unsqueeze(-1)
        out_idxs = out_idxs.unsqueeze(-1)
        # Now permute back
        out_vals = out_vals.permute(inv_perm)
        out_idxs = out_idxs.permute(inv_perm)
    else:
        # Permute back first, then we remove the dimension
        out_vals = out_vals.permute(inv_perm)
        out_idxs = out_idxs.permute(inv_perm)

    # Build the output
    # If out is provided, copy to out; otherwise, just return a namedtuple
    if out is not None:
        # out should be a tuple of two Tensors (max, max_indices)
        out[0].copy_(out_vals)
        out[1].copy_(out_idxs)
        return out
    else:
        return MaxReturn(out_vals, out_idxs)
