import torch
import triton
import triton.language as tl

@triton.jit
def scale_columns_kernel(
    output_ptr, input_ptr, scale_ptr,
    input_batch_stride, input_row_stride, input_col_stride,
    scale_batch_stride, scale_col_stride,
    n_batches, n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_col = tl.program_id(1)
    pid_row = tl.program_id(2)

    if pid_batch >= n_batches or pid_col >= n_cols:
        return

    row_start = pid_row * BLOCK_SIZE
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = row_offsets < n_rows

    input_batch_ptr = input_ptr + pid_batch * input_batch_stride
    scale_batch_ptr = scale_ptr + pid_batch * scale_batch_stride

    scale_val = tl.load(scale_batch_ptr + pid_col * scale_col_stride)

    input_col_ptr = input_batch_ptr + pid_col * input_col_stride
    output_col_ptr = output_batch_ptr + pid_col * input_col_stride

    input_ptrs = input_col_ptr + row_offsets * input_row_stride
    output_ptrs = output_col_ptr + row_offsets * input_row_stride

    rows = tl.load(input_ptrs, mask=mask, other=0.0)
    scaled_rows = rows * scale_val
    tl.store(output_ptrs, scaled_rows, mask=mask)

def scale_columns(V, L_pow):
    assert V.shape[:-2] == L_pow.shape[:-1], "Batch dimensions must match"
    assert V.size(-2) == V.size(-1) == L_pow.size(-1), "Matrix and scales dimensions must match"
    original_shape = V.shape
    V_flat = V.reshape(-1, *V.shape[-2:])
    L_pow_flat = L_pow.reshape(-1, L_pow.shape[-1])
    n_batches, n_rows, n_cols = V_flat.shape
    output = torch.empty_like(V_flat)

    BLOCK_SIZE = 128
    grid = (
        n_batches,
        n_cols,
        triton.cdiv(n_rows, BLOCK_SIZE),
    )
    scale_columns_kernel[grid](
        output,
        V_flat,
        L_pow_flat,
        V_flat.stride(0),
        V_flat.stride(1),
        V_flat.stride(2),
        L_pow_flat.stride(0),
        L_pow_flat.stride(1),
        n_batches,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    output = output.reshape(original_shape)
    return output

def matrix_power_eig(A, k, *, out=None):
    assert A.size(-1) == A.size(-2), "A must be a square matrix or batch of square matrices"
    L, V = torch.linalg.eig(A)
    L_pow = torch.pow(L, k)
    scaled_V = scale_columns(V, L_pow)
    V_inv = torch.linalg.inv(V)
    result = scaled_V @ V_inv
    if out is not None:
        if not out.is_complex() and result.is_complex():
            raise ValueError("out tensor must be complex dtype if result is complex")
        out.copy_(result)
        return out
    return result
