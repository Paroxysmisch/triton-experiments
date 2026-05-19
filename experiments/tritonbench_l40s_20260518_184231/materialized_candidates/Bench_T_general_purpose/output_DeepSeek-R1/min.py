import torch
import triton
import triton.language as tl

@triton.jit
def min_kernel(
    input_ptr,
    output_val_ptr,
    output_idx_ptr,
    row_size,
    input_row_stride,
    output_row_stride,
):
    row_idx = tl.program_id(0)
    input_row = input_ptr + row_idx * input_row_stride
    output_val_row = output_val_ptr + row_idx * output_row_stride
    output_idx_row = output_idx_ptr + row_idx * output_row_stride

    current_min = tl.load(input_row)
    current_idx = 0

    for i in range(1, row_size):
        val = tl.load(input_row + i)
        if val < current_min:
            current_min = val
            current_idx = i

    tl.store(output_val_row, current_min)
    tl.store(output_idx_row, current_idx)

def min(input: torch.Tensor, dim: int, keepdim: bool = False, *, out=None):
    assert input.dim() > 0, "input must have at least one dimension"
    dim = dim if dim >= 0 else dim + input.dim()
    assert 0 <= dim < input.dim(), "dim out of range"

    # Permute the target dimension to the end
    perm = list(range(input.dim()))
    perm.pop(dim)
    perm.append(dim)
    input_perm = input.permute(perm)
    input_contig = input_perm.contiguous()

    # Reshape to (num_rows, dim_size)
    dim_size = input_contig.size(-1)
    num_rows = input_contig.numel() // dim_size
    input_2d = input_contig.view(num_rows, dim_size)
    row_size = dim_size

    # Create output tensors
    if keepdim:
        min_vals_shape = (num_rows, 1)
        min_indices_shape = (num_rows, 1)
    else:
        min_vals_shape = (num_rows,)
        min_indices_shape = (num_rows,)
    min_vals = torch.empty(min_vals_shape, dtype=input.dtype, device=input.device)
    min_indices = torch.empty(min_indices_shape, dtype=torch.long, device=input.device)

    # Launch kernel
    grid = (num_rows,)
    min_kernel[grid](
        input_2d.data_ptr(),
        min_vals.data_ptr(),
        min_indices.data_ptr(),
        row_size,
        input_2d.stride(0),
        min_vals.stride(0)
    )

    # Reshape back to permuted shape
    output_shape = list(input_perm.shape[:-1]) + ([1] if keepdim else [])
    min_vals = min_vals.view(output_shape)
    min_indices = min_indices.view(output_shape)

    # Permute back if keepdim is True
    if keepdim:
        inv_perm = list(range(len(perm)))
        inv_perm.append(inv_perm.pop(dim))
        min_vals = min_vals.permute(inv_perm)
        min_indices = min_indices.permute(inv_perm)

    if out is not None:
        if not isinstance(out, tuple) or len(out) != 2:
            raise TypeError("out must be a tuple of two tensors")
        out[0].copy_(min_vals)
        out[1].copy_(min_indices)
        return out
    return (min_vals, min_indices)
