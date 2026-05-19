import torch
import triton
import triton.language as tl
import math

@triton.jit
def _softmax_kernel(
    output_ptr, input_ptr, mask_ptr,
    stride_om, stride_on, stride_ik,
    n_cols, BLOCK_SIZE: tl.constexpr,
    CAUSAL: tl.constexpr, LOG: tl.constexpr,
    MASK_TYPE: tl.constexpr,
):
    # Position of elements processed by this program
    row_idx = tl.program_id(0)
    
    # Compute memory offsets for this row
    row_start_ptr = input_ptr + row_idx * stride_on
    mask_row_start_ptr = mask_ptr + row_idx * stride_ik if MASK_TYPE != 0 else 0
    
    # Initialize pointers to load/store data
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask_ptrs = mask_row_start_ptr + col_offsets if MASK_TYPE != 0 else 0
    
    # Load input elements
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    
    # Apply mask if needed
    if MASK_TYPE != 0:
        mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=0)
        row = row + mask * -float('inf')
    
    # Apply causal mask if needed
    if CAUSAL:
        causal_mask = col_offsets >= (row_idx + 1)
        row = tl.where(causal_mask, -float('inf'), row)
    
    # Compute softmax
    row_max = tl.max(row, axis=0)
    row = row - row_max
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Apply log if needed
    if LOG:
        softmax_output = tl.log(softmax_output)
    
    # Write output
    output_ptrs = output_ptr + row_idx * stride_om + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
    ],
    key=['n_cols']
)
def softmax(input_tensor, mask=None, causal=False, log=False, mask_type=0):
    batch_size, seq_len, n_cols = input_tensor.shape
    
    # Allocate output
    output = torch.empty_like(input_tensor)
    
    # Calculate strides
    stride_om = output.stride(0)
    stride_on = input_tensor.stride(0)
    stride_ik = mask.stride(0) if mask is not None else 0
    
    # Compute grid
    grid = (batch_size * seq_len,)
    
    # Launch kernel
    _softmax_kernel[grid](
        output, input_tensor, mask if mask is not None else input_tensor,
        stride_om, stride_on, stride_ik,
        n_cols, BLOCK_SIZE=min(triton.next_power_of_2(n_cols), 512),
        CAUSAL=causal, LOG=log, MASK_TYPE=mask_type,
    )
    
    return output

@triton.jit
def _softmax_backward_kernel(
    grad_input_ptr, grad_output_ptr, output_ptr,
    stride_gim, stride_gon, stride_om,
    n_cols, BLOCK_SIZE: tl.constexpr,
    LOG: tl.constexpr
):
    # Position of elements processed by this program
    row_idx = tl.program_id(0)
    
    # Compute memory offsets for this row
    grad_output_row_ptr = grad_output_ptr + row_idx * stride_gon
    output_row_ptr = output_ptr + row_idx * stride_om
    
    # Initialize pointers to load/store data
    col_offsets = tl.arange(0, BLOCK_SIZE)
    grad_output_ptrs = grad_output_row_ptr + col_offsets
    output_ptrs = output_row_ptr + col_offsets
    
    # Load elements
    grad_output = tl.load(grad_output_ptrs, mask=col_offsets < n_cols)
    output = tl.load(output_ptrs, mask=col_offsets < n_cols)
    
    # Compute gradient
    if LOG:
        grad_input = grad_output - tl.exp(output) * tl.sum(grad_output, axis=0)
    else:
        sum_grad_output = tl.sum(grad_output * output, axis=0)
        grad_input = output * (grad_output - sum_grad_output)
    
    # Store gradient
    grad_input_ptrs = grad_input_ptr + row_idx * stride_gim + col_offsets
    tl.store(grad_input_ptrs, grad_input, mask=col_offsets < n_cols)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
    ],
    key=['n_cols']
)
def softmax_backward(grad_output, output, log=False):
    batch_size, seq_len, n_cols = grad_output.shape
    
    # Allocate output
    grad_input = torch.empty_like(grad_output)
    
    # Calculate strides
    stride_gim = grad_input.stride(0)
    stride_gon = grad_output.stride(0)
    stride_om = output.stride(0)
    
    # Compute grid
    grid = (batch_size * seq_len,)
    
    # Launch kernel
    _softmax_backward_kernel[grid](
        grad_input, grad_output, output,
        stride_gim, stride_gon, stride_om,
        n_cols, BLOCK_SIZE=min(triton.next_power_of_2(n_cols), 512),
        LOG=log,
    )
    
    return grad_input
