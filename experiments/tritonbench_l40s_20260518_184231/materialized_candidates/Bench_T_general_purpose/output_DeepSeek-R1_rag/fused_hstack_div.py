import triton
import triton.language as tl
import torch
import math

# Triton kernel for division with tensor divisor and optional rounding
@triton.jit
def div_tensor_kernel(
    input_ptr,
    div_ptr,
    output_ptr,
    input_rows,
    input_cols,
    div_rows,
    div_cols,
    rounding_mode: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    
    row_offsets = pid_row * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    col_offsets = pid_col * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    
    row_mask = row_offsets < input_rows
    col_mask = col_offsets < input_cols
    full_mask = row_mask[:, None] & col_mask[None, :]
    
    # Calculate input and divisor pointers with broadcasting
    input_idx = row_offsets[:, None] * input_cols + col_offsets[None, :]
    div_row = row_offsets % div_rows
    div_col = col_offsets % div_cols
    div_idx = div_row[:, None] * div_cols + div_col[None, :]
    
    input_val = tl.load(input_ptr + input_idx, mask=full_mask, other=0)
    div_val = tl.load(div_ptr + div_idx, mask=full_mask, other=1)  # Avoid division by zero
    
    result = input_val / div_val
    
    if rounding_mode == 'trunc':
        result = tl.math.trunc(result)
    elif rounding_mode == 'floor':
        result = tl.math.floor(result)
    
    tl.store(output_ptr + input_idx, result, mask=full_mask)

# Triton kernel for division with scalar divisor and optional rounding
@triton.jit
def div_scalar_kernel(
    input_ptr,
    div_scalar,
    output_ptr,
    input_rows,
    input_cols,
    rounding_mode: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    
    row_offsets = pid_row * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    col_offsets = pid_col * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    
    row_mask = row_offsets < input_rows
    col_mask = col_offsets < input_cols
    full_mask = row_mask[:, None] & col_mask[None, :]
    
    input_idx = row_offsets[:, None] * input_cols + col_offsets[None, :]
    input_val = tl.load(input_ptr + input_idx, mask=full_mask, other=0)
    
    result = input_val / div_scalar
    
    if rounding_mode == 'trunc':
        result = tl.math.trunc(result)
    elif rounding_mode == 'floor':
        result = tl.math.floor(result)
    
    tl.store(output_ptr + input_idx, result, mask=full_mask)

def fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None):
    # Step 1: Perform horizontal stacking
    stacked = torch.hstack(tensors)
    
    # Step 2: Prepare output tensor
    if out is None:
        out = torch.empty_like(stacked)
    else:
        assert out.shape == stacked.shape, "Output tensor shape mismatch"
    
    # Step 3: Configure kernel launch parameters
    rows, cols = stacked.shape if len(stacked.shape) > 1 else (1, stacked.shape[0])
    BLOCK_SIZE_ROW = 16
    BLOCK_SIZE_COL = 16
    grid = (
        triton.cdiv(rows, BLOCK_SIZE_ROW),
        triton.cdiv(cols, BLOCK_SIZE_COL),
    )
    
    # Step 4: Launch appropriate kernel based on divisor type
    if isinstance(divisor, torch.Tensor):
        assert divisor.dim() <= 2, "Divisor tensor must be 1D or 2D"
        div_rows = divisor.shape[0] if divisor.dim() > 1 else 1
        div_cols = divisor.shape[-1]
        div_tensor_kernel[grid](
            stacked,
            divisor,
            out,
            rows,
            cols,
            div_rows,
            div_cols,
            rounding_mode,
            BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
            BLOCK_SIZE_COL=BLOCK_SIZE_COL,
        )
    else:  # Scalar divisor
        div_scalar_kernel[grid](
            stacked,
            divisor,
            out,
            rows,
            cols,
            rounding_mode,
            BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
            BLOCK_SIZE_COL=BLOCK_SIZE_COL,
        )
    
    return out
