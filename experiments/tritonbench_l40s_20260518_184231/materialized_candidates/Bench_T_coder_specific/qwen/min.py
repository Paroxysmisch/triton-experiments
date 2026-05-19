import torch
import triton
import triton.language as tl

# Define the Triton kernel
min_kernel = triton.compile(min_kernel, signature='(x,min,min_idx,n_rows,n_cols,stride_row,stride_col)->(),()', num_warps=4)

def min(input, dim, keepdim=False, out=None):
    n_rows, n_cols = input.shape
    stride_row = input.stride(0)
    stride_col = input.stride(1)

    # Allocate output tensors
    if out is None:
        min_tensor = torch.empty(n_rows, device=input.device, dtype=input.dtype)
        min_idx_tensor = torch.empty(n_rows, device=input.device, dtype=torch.long)
    else:
        min_tensor, min_idx_tensor = out

    # Launch the Triton kernel
    grid = lambda meta: (
        tl.cdiv(n_rows, meta['block_size']),
        n_cols // meta['block_size'],
    )
    min_kernel[grid](input.data_ptr(), min_tensor.data_ptr(), min_idx_tensor.data_ptr(), n_rows, n_cols, stride_row, stride_col, block=(meta['block_size'], 1, 1))

    # Squeeze the output if keepdim is False
    if not keepdim:
        min_tensor = min_tensor.unsqueeze(dim)
        min_idx_tensor = min_idx_tensor.unsqueeze(dim)

    return min_tensor, min_idx_tensor
