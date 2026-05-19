import triton
import triton.language as tl
import torch

@triton.jit
def _sgmv_expand_slice_kernel(
    lora_weights_ptr, lora_weights_stride, 
    input_ptr, input_stride, 
    output_ptr, output_stride, 
    row_indices_ptr, row_indices_stride,
    col_indices_ptr, col_indices_stride,
    num_rows, num_cols, nnz,
    BLOCKSIZE_ROW: tl.constexpr, BLOCKSIZE_COL: tl.constexpr
):
    # Program IDs
    pid = tl.program_id(axis=0)
    
    # Compute row and column index
    row_idx = tl.load(row_indices_ptr + pid * row_indices_stride)
    col_idx = tl.load(col_indices_ptr + pid * col_indices_stride)
    
    # Load LoRA weights and input vector
    lora_weights = tl.load(lora_weights_ptr + row_idx * lora_weights_stride)
    input_val = tl.load(input_ptr + col_idx * input_stride)
    
    # Perform the multiplication
    result = lora_weights * input_val
    
    # Accumulate results
    acc = tl.zeros([BLOCKSIZE_ROW, BLOCKSIZE_COL], dtype=tl.float32)
    acc += result
    
    # Store the result in the output
    tl.store(output_ptr + row_idx * output_stride, acc)


def _sgmv_expand_slice(
    lora_weights: torch.Tensor,
    input: torch.Tensor,
    row_indices: torch.Tensor,
    col_indices: torch.Tensor,
    output: torch.Tensor,
    block_size_row: int = 128,
    block_size_col: int = 128
):
    # Determine the number of non-zero elements
    nnz = row_indices.shape[0]
    
    # Launch the Triton kernel
    grid = (nnz,)
    _sgmv_expand_slice_kernel[grid](
        lora_weights_ptr=lora_weights.data_ptr(),
        lora_weights_stride=lora_weights.stride(0),
        input_ptr=input.data_ptr(),
        input_stride=input.stride(0),
        output_ptr=output.data_ptr(),
        output_stride=output.stride(0),
        row_indices_ptr=row_indices.data_ptr(),
        row_indices_stride=row_indices.stride(0),
        col_indices_ptr=col_indices.data_ptr(),
        col_indices_stride=col_indices.stride(0),
        num_rows=lora_weights.shape[0],
        num_cols=input.shape[0],
        nnz=nnz,
        BLOCKSIZE_ROW=block_size_row,
        BLOCKSIZE_COL=block_size_col
    )
