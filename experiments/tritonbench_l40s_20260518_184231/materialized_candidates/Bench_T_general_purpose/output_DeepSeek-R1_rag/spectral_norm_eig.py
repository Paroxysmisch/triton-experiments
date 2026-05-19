import triton
import triton.language as tl
import torch

@triton.jit
def max_abs_reduce_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    max_val = tl.max(row, axis=0)
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr, max_val)

def max_abs_reduce(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    y = torch.empty(n_rows, dtype=x.dtype, device=x.device)
    max_abs_reduce_kernel[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y

def spectral_norm_eig(A, *, out=None):
    if not (A.shape[-1] == A.shape[-2]):
        raise ValueError("Input must be a square matrix or batch of square matrices.")
    eigs = torch.linalg.eigvals(A)
    abs_eigs = torch.abs(eigs)
    original_shape = abs_eigs.shape[:-1]
    flattened_abs_eigs = abs_eigs.reshape(-1, abs_eigs.size(-1))
    norms_flattened = max_abs_reduce(flattened_abs_eigs)
    norms = norms_flattened.reshape(original_shape)
    if out is not None:
        out.copy_(norms)
        return out
    return norms
