import torch
import triton
import triton.language as tl
import math

@triton.jit
def softmax_kernel_forward(
    output_ptr, input_ptr,
    stride_om, stride_on,  # output strides
    stride_im, stride_in,  # input strides
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID gives position in the grid
    row_idx = tl.program_id(0)
    
    # Compute pointers to input and output rows
    input_row_ptr = input_ptr + row_idx * stride_im
    output_row_ptr = output_ptr + row_idx * stride_om
    
    # Create offsets for the columns
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_row_ptr + col_offsets * stride_in
    
    # Load input row with mask for handling non-power-of-2 sizes
    row_mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=row_mask, other=-float('inf'))
    
    # Numerical stability: subtract maximum
    row_max = tl.max(row, axis=0)
    row = row - row_max
    
    # Compute exponentials and sum
    exp_row = tl.exp(row)
    sum_exp = tl.sum(exp_row, axis=0)
    
    # Normalize
    softmax_output = exp_row / sum_exp
    
    # Store results
    output_ptrs = output_row_ptr + col_offsets * stride_on
    tl.store(output_ptrs, softmax_output, mask=row_mask)

@triton.jit
def softmax_kernel_backward(
    grad_output_ptr, grad_input_ptr, output_ptr,
    stride_gom, stride_gon,  # grad output strides
    stride_gim, stride_gin,  # grad input strides
    stride_om, stride_on,    # output strides
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    
    # Compute pointers
    grad_output_row_ptr = grad_output_ptr + row_idx * stride_gom
    grad_input_row_ptr = grad_input_ptr + row_idx * stride_gim
    output_row_ptr = output_ptr + row_idx * stride_om
    
    # Create offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = col_offsets < n_cols
    
    # Load grad_output and output
    grad_output_ptrs = grad_output_row_ptr + col_offsets * stride_gon
    output_ptrs = output_row_ptr + col_offsets * stride_on
    
    grad_output = tl.load(grad_output_ptrs, mask=row_mask, other=0.0)
    output = tl.load(output_ptrs, mask=row_mask, other=0.0)
    
    # Compute gradient
    sum_grad_output = tl.sum(grad_output * output, axis=0)
    grad_input = output * (grad_output - sum_grad_output)
    
    # Store gradient
    grad_input_ptrs = grad_input_row_ptr + col_offsets * stride_gin
    tl.store(grad_input_ptrs, grad_input, mask=row_mask)

class TritonSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save input shape and flatten if needed
        orig_shape = x.shape
        if len(orig_shape) > 2:
            x = x.view(-1, orig_shape[-1])
        
        # Get dimensions
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        
        # Allocate output
        output = torch.empty_like(x)
        
        # Compute number of warps based on problem size
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16
            
        # Launch kernel
        softmax_kernel_forward[(n_rows,)](
            output, x,
            output.stride(0), output.stride(1),
            x.stride(0), x.stride(1),
            n_cols,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Save for backward
        ctx.save_for_backward(output)
        ctx.orig_shape = orig_shape
        
        # Reshape output if needed
        if len(orig_shape) > 2:
            output = output.view(orig_shape)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        
        # Reshape if needed
        if len(orig_shape) > 2:
            grad_output = grad_output.view(-1, orig_shape[-1])
            output = output.view(-1, orig_shape[-1])
        
        # Get dimensions
        n_rows, n_cols = output.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        
        # Allocate gradient
        grad_input = torch.empty_like(output)
        
        # Compute number of warps
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16
            
        # Launch backward kernel
        softmax_kernel_backward[(n_rows,)](
            grad_output, grad_input, output,
            grad_output.stride(0), grad_output.stride(1),
            grad_input.stride(0), grad_input.stride(1),
            output.stride(0), output.stride(1),
            n_cols,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Reshape gradient if needed
        if len(orig_shape) > 2:
            grad_input = grad_input.view(orig_shape)
            
        return grad_input

def triton_softmax(x):
    return TritonSoftmax.apply(x)
