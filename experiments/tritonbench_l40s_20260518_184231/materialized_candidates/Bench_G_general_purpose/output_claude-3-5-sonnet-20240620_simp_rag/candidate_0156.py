import torch
import triton
import triton.language as tl

@triton.jit
def _softmax(
    output_ptr, input_ptr, mask_ptr,
    batch_stride, row_stride, col_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    IS_LOG: tl.constexpr,
    CAUSAL: tl.constexpr,
    MASK_TYPE: tl.constexpr,  # 0: no mask, 1: regular mask, 2: causal mask
):
    # Program ID
    row_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    # Compute pointers
    batch_offset = batch_idx * batch_stride
    row_offset = row_idx * row_stride
    input_ptr = input_ptr + batch_offset + row_offset
    output_ptr = output_ptr + batch_offset + row_offset
    
    # Load input row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + col_offsets * col_stride
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    
    # Apply mask if needed
    if MASK_TYPE == 1 and mask_ptr is not None:
        mask = tl.load(mask_ptr + batch_offset + row_offset + col_offsets)
        row = row + mask * -float('inf')
    elif MASK_TYPE == 2 and CAUSAL:
        causal_mask = col_offsets >= (row_idx + 1)
        row = row + causal_mask * -float('inf')
    
    # Compute softmax
    row_max = tl.max(row, axis=0)
    row = row - row_max
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Apply log if needed
    if IS_LOG:
        softmax_output = tl.log(softmax_output)
    
    # Store output
    output_ptrs = output_ptr + col_offsets * col_stride
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

@triton.jit
def _softmax_backward(
    grad_input_ptr, grad_output_ptr, softmax_output_ptr,
    batch_stride, row_stride, col_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    row_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    
    # Compute pointers
    batch_offset = batch_idx * batch_stride
    row_offset = row_idx * row_stride
    grad_output_ptr = grad_output_ptr + batch_offset + row_offset
    softmax_output_ptr = softmax_output_ptr + batch_offset + row_offset
    grad_input_ptr = grad_input_ptr + batch_offset + row_offset
    
    # Load data
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    grad_output_ptrs = grad_output_ptr + col_offsets * col_stride
    softmax_output_ptrs = softmax_output_ptr + col_offsets * col_stride
    
    grad_output = tl.load(grad_output_ptrs, mask=mask)
    softmax_output = tl.load(softmax_output_ptrs, mask=mask)
    
    # Compute gradient
    sum_grad_times_output = tl.sum(grad_output * softmax_output, axis=0)
    grad_input = softmax_output * (grad_output - sum_grad_times_output)
    
    # Store gradient
    grad_input_ptrs = grad_input_ptr + col_offsets * col_stride
    tl.store(grad_input_ptrs, grad_input, mask=mask)

def get_configs():
    return [
        triton.Config({'BLOCK_SIZE': 64, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 128, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 256, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE': 512, 'num_warps': 8}),
    ]

@triton.autotune(configs=get_configs())
def softmax(x, mask=None, is_log=False, causal=False, mask_type=0):
    batch, rows, cols = x.shape
    
    # Allocate output
    output = torch.empty_like(x)
    
    # Launch kernel
    grid = (rows, batch)
    _softmax[grid](
        output, x, mask,
        x.stride(0), x.stride(1), x.stride(2),
        cols,
        BLOCK_SIZE=triton.next_power_of_2(cols),
        IS_LOG=is_log,
        CAUSAL=causal,
        MASK_TYPE=mask_type,
    )
    return output

@triton.autotune(configs=get_configs())
def softmax_backward(grad_output, softmax_output):
    batch, rows, cols = grad_output.shape
    
    # Allocate output
    grad_input = torch.empty_like(grad_output)
    
    # Launch kernel
    grid = (rows, batch)
    _softmax_backward[grid](
        grad_input, grad_output, softmax_output,
        grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
        cols,
        BLOCK_SIZE=triton.next_power_of_2(cols),
    )
    return grad_input
