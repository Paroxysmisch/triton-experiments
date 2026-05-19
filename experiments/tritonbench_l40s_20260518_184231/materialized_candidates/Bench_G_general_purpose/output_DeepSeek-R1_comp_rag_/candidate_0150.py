import torch
import triton
import triton.language as tl
from triton.compiler import AttrsDescriptor, ASTSource
import triton.compiler as tc

# Forward Kernel
@triton.jit
def _softmax(
    output_ptr,
    input_ptr,
    input_batch_stride,
    input_seq_stride,
    input_dim_stride,
    output_batch_stride,
    output_seq_stride,
    output_dim_stride,
    B,
    M,
    N,
    mask_type: tl.constexpr,
    is_log: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_step = tl.num_programs(0)
    
    for row_offset in tl.range(0, B * M, row_step, num_stages=num_stages):
        current_row = row_idx * num_stages + row_offset
        if current_row >= B * M:
            break
        
        b = current_row // M
        m = current_row % M
        
        input_row_start = input_ptr + b * input_batch_stride + m * input_seq_stride
        output_row_start = output_ptr + b * output_batch_stride + m * output_seq_stride
        
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = input_row_start + col_offsets * input_dim_stride
        mask = col_offsets < N
        
        if mask_type == 'CAUSAL':
            causal_mask = col_offsets <= m
            mask = mask & causal_mask
        
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_max = tl.max(row, axis=0)
        row_minus_max = row - row_max
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        
        if is_log:
            output = row_minus_max - tl.log(denominator)
        else:
            output = numerator / denominator
        
        output_ptrs = output_row_start + col_offsets * output_dim_stride
        tl.store(output_ptrs, output, mask=mask)

# Backward Kernel
@triton.jit
def _softmax_backward(
    grad_input_ptr,
    grad_output_ptr,
    output_ptr,
    grad_input_batch_stride,
    grad_input_seq_stride,
    grad_input_dim_stride,
    grad_output_batch_stride,
    grad_output_seq_stride,
    grad_output_dim_stride,
    output_batch_stride,
    output_seq_stride,
    output_dim_stride,
    B,
    M,
    N,
    mask_type: tl.constexpr,
    is_log: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_step = tl.num_programs(0)
    
    for row_offset in tl.range(0, B * M, row_step, num_stages=num_stages):
        current_row = row_idx * num_stages + row_offset
        if current_row >= B * M:
            break
        
        b = current_row // M
        m = current_row % M
        
        grad_output_row = grad_output_ptr + b * grad_output_batch_stride + m * grad_output_seq_stride
        output_row = output_ptr + b * output_batch_stride + m * output_seq_stride
        grad_input_row = grad_input_ptr + b * grad_input_batch_stride + m * grad_input_seq_stride
        
        col_offsets = tl.arange(0, BLOCK_SIZE)
        grad_output_ptrs = grad_output_row + col_offsets * grad_output_dim_stride
        output_ptrs = output_row + col_offsets * output_dim_stride
        grad_input_ptrs = grad_input_row + col_offsets * grad_input_dim_stride
        
        mask = col_offsets < N
        if mask_type == 'CAUSAL':
            causal_mask = col_offsets <= m
            mask = mask & causal_mask
        
        grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0)
        output = tl.load(output_ptrs, mask=mask, other=0.0)
        
        if is_log:
            exp_output = tl.exp(output)
            sum_grad = tl.sum(grad_output, axis=0)
            grad_input = grad_output - exp_output * sum_grad
        else:
            sum_term = tl.sum(output * grad_output, axis=0)
            grad_input = output * (grad_output - sum_term)
        
        tl.store(grad_input_ptrs, grad_input, mask=mask)

# Host Function for Forward Pass
def softmax(x, mask_type='NONE', is_log=False):
    assert x.dim() == 3, "Input must be 3D tensor"
    B, M, N = x.shape
    output = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    total_rows = B * M
    
    grid = (triton.cdiv(total_rows, 4),)  # Using 4 stages
    
    _softmax[grid](
        output, x,
        x.stride(0), x.stride(1), x.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        B, M, N,
        mask_type, is_log,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=4,
        num_warps=8,
    )
    return output

# Host Function for Backward Pass
def softmax_backward(grad_output, output, mask_type='NONE', is_log=False):
    assert grad_output.dim() == 3 and output.dim() == 3, "Inputs must be 3D"
    B, M, N = grad_output.shape
    grad_input = torch.empty_like(grad_output)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    total_rows = B * M
    
    grid = (triton.cdiv(total_rows, 4),)
    
    _softmax_backward[grid](
        grad_input, grad_output, output,
        grad_input.stride(0), grad_input.stride(1), grad_input.stride(2),
        grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        B, M, N,
        mask_type, is_log,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=4,
        num_warps=8,
    )
    return grad_input
