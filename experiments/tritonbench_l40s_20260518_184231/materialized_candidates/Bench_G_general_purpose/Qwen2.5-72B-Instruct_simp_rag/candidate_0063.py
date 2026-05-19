import triton
import triton.language as tl
import torch
from typing import Optional, Tuple

@triton.jit
def _sgmv_expand_slice_kernel(
    batch_size: tl.constexpr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    n_lora: tl.constexpr,
    block_size: tl.constexpr,
    values_ptr,
    values_batch_stride,
    values_nnz_stride,
    values_row_block_stride,
    values_col_block_stride,
    lora_values_ptr,
    lora_values_batch_stride,
    lora_values_nnz_stride,
    lora_values_row_block_stride,
    lora_values_col_block_stride,
    crow_indices_ptr,
    crow_indices_batch_stride,
    crow_indices_stride,
    col_indices_ptr,
    col_indices_batch_stride,
    col_indices_stride,
    input_ptr,
    input_batch_stride,
    input_row_stride,
    input_col_stride,
    output_ptr,
    output_batch_stride,
    output_row_stride,
    output_col_stride,
    acc_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_id = pid % (n_rows // block_size)
    batch_id = pid // (n_rows // block_size)

    block_start_row = block_id * block_size
    block_end_row = block_start_row + block_size

    crow_start = tl.load(crow_indices_ptr + batch_id * crow_indices_batch_stride + block_id * crow_indices_stride)
    crow_end = tl.load(crow_indices_ptr + batch_id * crow_indices_batch_stride + (block_id + 1) * crow_indices_stride)

    for nnz in range(crow_start, crow_end):
        col_id = tl.load(col_indices_ptr + batch_id * col_indices_batch_stride + nnz * col_indices_stride)
        value = tl.load(values_ptr + batch_id * values_batch_stride + nnz * values_nnz_stride)

        for i in range(block_size):
            if block_start_row + i < n_rows:
                input_val = tl.load(input_ptr + batch_id * input_batch_stride + (block_start_row + i) * input_row_stride + col_id * input_col_stride)
                output_val = tl.load(output_ptr + batch_id * output_batch_stride + (block_start_row + i) * output_row_stride + col_id * output_col_stride)
                output_val += value * input_val
                tl.store(output_ptr + batch_id * output_batch_stride + (block_start_row + i) * output_row_stride + col_id * output_col_stride, output_val)

        if n_lora > 0:
            for lora_id in range(n_lora):
                lora_value = tl.load(lora_values_ptr + batch_id * lora_values_batch_stride + nnz * lora_values_nnz_stride + lora_id * lora_values_row_block_stride)
                for i in range(block_size):
                    if block_start_row + i < n_rows:
                        input_val = tl.load(input_ptr + batch_id * input_batch_stride + (block_start_row + i) * input_row_stride + col_id * input_col_stride)
                        output_val = tl.load(output_ptr + batch_id * output_batch_stride + (block_start_row + i) * output_row_stride + col_id * output_col_stride)
                        output_val += lora_value * input_val
                        tl.store(output_ptr + batch_id * output_batch_stride + (block_start_row + i) * output_row_stride + col_id * output_col_stride, output_val)

def _sgmv_expand_slice(
    batch_size: int,
    n_rows: int,
    n_cols: int,
    n_lora: int,
    block_size: int,
    values: torch.Tensor,
    lora_values: Optional[torch.Tensor],
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    input: torch.Tensor,
    output: torch.Tensor,
    max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
):
    assert values.is_cuda and crow_indices.is_cuda and col_indices.is_cuda and input.is_cuda and output.is_cuda
    assert values.dtype == input.dtype == output.dtype
    assert crow_indices.dtype == col_indices.dtype == torch.int32

    if max_grid is None:
        max_grid = (1024, 1024, 1024)

    grid = (batch_size * (n_rows // block_size), 1, 1)

    _sgmv_expand_slice_kernel[grid](
        batch_size,
        n_rows,
        n_cols,
        n_lora,
        block_size,
        values,
        values.stride(0),
        values.stride(1),
        values.stride(2),
        values.stride(3),
        lora_values if lora_values is not None else values.new_zeros(0),
        lora_values.stride(0) if lora_values is not None else 0,
        lora_values.stride(1) if lora_values is not None else 0,
        lora_values.stride(2) if lora_values is not None else 0,
        lora_values.stride(3) if lora_values is not None else 0,
        crow_indices,
        crow_indices.stride(0),
        crow_indices.stride(1),
        col_indices,
        col_indices.stride(0),
        col_indices.stride(1),
        input,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        output,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        acc_dtype=values.dtype,
        allow_tf32=True,
    )

# Example usage
if __name__ == "__main__":
    batch_size = 2
    n_rows = 128
    n_cols = 64
    n_lora = 2
    block_size = 16

    values = torch.randn(batch_size, n_rows // block_size, block_size, block_size, device='cuda')
    lora_values = torch.randn(batch_size, n_rows // block_size, block_size, n_lora, device='cuda')
    crow_indices = torch.randint(0, n_rows, (batch_size, n_rows // block_size + 1), device='cuda', dtype=torch.int32)
    col_indices = torch.randint(0, n_cols, (batch_size, n_rows // block_size * block_size), device='cuda', dtype=torch.int32)
    input = torch.randn(batch_size, n_rows, n_cols, device='cuda')
    output = torch.zeros_like(input)

    _sgmv_expand_slice(
        batch_size,
        n_rows,
        n_cols,
        n_lora,
        block_size,
        values,
        lora_values,
        crow_indices,
        col_indices,
        input,
        output,
    )

    print(output)
