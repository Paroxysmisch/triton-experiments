import triton
import triton.language as tl
import torch

@triton.jit
def _sgmv_expand_slice_kernel(
    input_ptr,
    lora_weights_ptr,
    lora_indices_ptr,
    output_ptr,
    batch_size,
    input_cols,
    output_cols,
    input_batch_stride,
    input_col_stride,
    lora_weights_lora_stride,
    lora_weights_row_stride,
    lora_weights_col_stride,
    lora_indices_stride,
    output_batch_stride,
    output_col_stride,
    BLOCK_COL: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_col_block = tl.program_id(1)
    
    if pid_batch >= batch_size:
        return
    
    lora_idx = tl.load(lora_indices_ptr + pid_batch * lora_indices_stride)
    
    input_start = input_ptr + pid_batch * input_batch_stride
    lora_start = lora_weights_ptr + lora_idx * lora_weights_lora_stride
    
    col_offset = pid_col_block * BLOCK_COL
    col_indices = col_offset + tl.arange(0, BLOCK_COL)
    col_mask = col_indices < output_cols
    
    acc = tl.zeros((BLOCK_COL,), dtype=ACC_TYPE)
    
    for block_row in range(0, input_cols, BLOCK_ROW):
        row_indices = block_row + tl.arange(0, BLOCK_ROW)
        row_mask = row_indices < input_cols
        
        input_vals = tl.load(input_start + row_indices * input_col_stride, mask=row_mask, other=0.0)
        
        lora_ptrs = lora_start + row_indices[:, None] * lora_weights_row_stride + col_indices[None, :] * lora_weights_col_stride
        lora_weights = tl.load(lora_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        
        acc += tl.sum(input_vals[:, None] * lora_weights, axis=0)
    
    output_ptr_batch = output_ptr + pid_batch * output_batch_stride
    output_ptrs = output_ptr_batch + col_indices * output_col_stride
    tl.store(output_ptrs, acc.to(output_ptr.dtype.element_ty), mask=col_mask)

def _sgmv_expand_slice(
    input: torch.Tensor,
    lora_weights: torch.Tensor,
    lora_indices: torch.Tensor,
    output: torch.Tensor,
    BLOCK_ROW: int = 64,
    BLOCK_COL: int = 128,
):
    assert input.is_contiguous()
    assert lora_weights.is_contiguous()
    assert lora_indices.is_contiguous()
    assert output.is_contiguous()
    
    batch_size, input_cols = input.shape
    num_loras, lora_input_cols, output_cols = lora_weights.shape
    assert input_cols == lora_input_cols, "Input columns must match LoRA input dimension"
    assert output.shape == (batch_size, output_cols), "Output shape mismatch"
    
    device = input.device
    assert device == lora_weights.device == output.device == lora_indices.device
    
    grid = (batch_size, triton.cdiv(output_cols, BLOCK_COL))
    
    ACC_TYPE = tl.float32 if input.dtype == torch.float32 else tl.float32
    
    _sgmv_expand_slice_kernel[grid](
        input_ptr=input,
        lora_weights_ptr=lora_weights,
        lora_indices_ptr=lora_indices,
        output_ptr=output,
        batch_size=batch_size,
        input_cols=input_cols,
        output_cols=output_cols,
        input_batch_stride=input.stride(0),
        input_col_stride=input.stride(1),
        lora_weights_lora_stride=lora_weights.stride(0),
        lora_weights_row_stride=lora_weights.stride(1),
        lora_weights_col_stride=lora_weights.stride(2),
        lora_indices_stride=lora_indices.stride(0),
        output_batch_stride=output.stride(0),
        output_col_stride=output.stride(1),
        BLOCK_COL=BLOCK_COL,
        BLOCK_ROW=BLOCK_ROW,
        ACC_TYPE=ACC_TYPE,
    )
