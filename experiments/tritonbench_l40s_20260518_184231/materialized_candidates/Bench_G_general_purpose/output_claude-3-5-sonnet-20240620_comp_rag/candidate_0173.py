import torch
import triton
import triton.language as tl
from torch.autograd import Function

@triton.jit
def log_softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Calculate pointers for the current row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Load row data
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    
    # Compute row maximum for numerical stability
    row_max = tl.max(row, axis=0)
    row_minus_max = row - row_max
    
    # Compute exponentials and their sum
    exp_values = tl.exp(row_minus_max)
    sum_exp = tl.sum(exp_values, axis=0)
    log_sum_exp = tl.log(sum_exp)
    
    # Compute log softmax
    log_softmax_output = row_minus_max - log_sum_exp
    
    # Store results
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, log_softmax_output, mask=col_offsets < n_cols)

@triton.jit
def log_softmax_backward_kernel(
    grad_output_ptr, output_ptr, grad_input_ptr,
    row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    # Calculate pointers
    col_offsets = tl.arange(0, BLOCK_SIZE)
    grad_output_row_ptr = grad_output_ptr + row_idx * row_stride
    output_row_ptr = output_ptr + row_idx * row_stride
    
    # Load data
    grad_output = tl.load(grad_output_row_ptr + col_offsets, 
                         mask=col_offsets < n_cols, other=0.0)
    output = tl.load(output_row_ptr + col_offsets,
                    mask=col_offsets < n_cols, other=0.0)
    
    # Compute gradient
    exp_output = tl.exp(output)
    sum_grad_output = tl.sum(grad_output, axis=0)
    grad_input = grad_output - exp_output * sum_grad_output
    
    # Store results
    grad_input_row_ptr = grad_input_ptr + row_idx * row_stride
    tl.store(grad_input_row_ptr + col_offsets, grad_input, 
             mask=col_offsets < n_cols)

class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, input, dim=-1):
        if dim != -1 and dim != input.dim() - 1:
            input = input.transpose(dim, -1)
        
        # Ensure contiguous memory layout
        input = input.contiguous()
        output = torch.empty_like(input)
        
        # Calculate dimensions
        *batch_dims, n_cols = input.shape
        n_rows = input.numel() // n_cols
        
        # Calculate optimal block size
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        BLOCK_SIZE = min(BLOCK_SIZE, 1024)
        
        # Launch kernel
        log_softmax_kernel[(n_rows,)](
            input.data_ptr(),
            output.data_ptr(),
            input.stride(-2) if input.dim() > 1 else 0,
            output.stride(-2) if output.dim() > 1 else 0,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        ctx.save_for_backward(output)
        ctx.dim = dim
        
        if dim != -1 and dim != input.dim() - 1:
            output = output.transpose(dim, -1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        dim = ctx.dim
        
        if dim != -1 and dim != grad_output.dim() - 1:
            grad_output = grad_output.transpose(dim, -1)
            output = output.transpose(dim, -1)
        
        grad_output = grad_output.contiguous()
        grad_input = torch.empty_like(grad_output)
        
        *batch_dims, n_cols = grad_output.shape
        n_rows = grad_output.numel() // n_cols
        
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        BLOCK_SIZE = min(BLOCK_SIZE, 1024)
        
        log_softmax_backward_kernel[(n_rows,)](
            grad_output.data_ptr(),
            output.data_ptr(),
            grad_input.data_ptr(),
            grad_output.stride(-2) if grad_output.dim() > 1 else 0,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        if dim != -1 and dim != grad_output.dim() - 1:
            grad_input = grad_input.transpose(dim, -1)
        
        return grad_input, None

def log_softmax(input, dim=-1):
    return LogSoftmax.apply(input, dim)
