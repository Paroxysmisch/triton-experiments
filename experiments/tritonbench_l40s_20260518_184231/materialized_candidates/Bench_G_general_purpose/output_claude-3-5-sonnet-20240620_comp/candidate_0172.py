import torch
import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(
    output_ptr, input_ptr, M, N,
    stride_om, stride_on,
    stride_im, stride_in,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Position of blocks
    pid = tl.program_id(0)
    
    # Row index
    row_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Column indices
    col_idx = tl.arange(0, BLOCK_N)
    
    # Load input elements
    row_mask = row_idx < M
    col_mask = col_idx < N
    mask = row_mask[:, None] & col_mask[None, :]
    
    # Compute row offsets
    row_offs = row_idx[:, None] * stride_im
    col_offs = col_idx[None, :] * stride_in
    offs = row_offs + col_offs
    
    # Load input elements
    x = tl.load(input_ptr + offs, mask=mask, other=-float('inf'))
    
    # Row-wise maximum for numerical stability
    row_max = tl.max(x, axis=1)[:, None]
    
    # Compute exponentials
    x_stable = x - row_max
    numerator = tl.exp(x_stable)
    
    # Compute row-wise sum of exponentials
    denominator = tl.sum(numerator, axis=1)[:, None]
    
    # Compute log softmax
    log_softmax = x_stable - tl.log(denominator)
    
    # Store result
    output_offs = row_idx[:, None] * stride_om + col_idx[None, :] * stride_on
    tl.store(output_ptr + output_offs, log_softmax, mask=mask)

@triton.jit
def log_softmax_backward_kernel(
    grad_input_ptr, grad_output_ptr, output_ptr, M, N,
    stride_gim, stride_gin,
    stride_gom, stride_gon,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Position of blocks
    pid = tl.program_id(0)
    
    # Row index
    row_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Column indices
    col_idx = tl.arange(0, BLOCK_N)
    
    # Load masks
    row_mask = row_idx < M
    col_mask = col_idx < N
    mask = row_mask[:, None] & col_mask[None, :]
    
    # Compute offsets
    row_offs = row_idx[:, None] * stride_om
    col_offs = col_idx[None, :] * stride_on
    offs = row_offs + col_offs
    
    # Load output and grad_output
    output = tl.load(output_ptr + offs, mask=mask, other=0)
    grad_output_offs = row_idx[:, None] * stride_gom + col_idx[None, :] * stride_gon
    grad_output = tl.load(grad_output_ptr + grad_output_offs, mask=mask, other=0)
    
    # Compute sum of gradients
    sum_grad = tl.sum(grad_output, axis=1)[:, None]
    
    # Compute gradients
    grad_input = grad_output - tl.exp(output) * sum_grad
    
    # Store gradients
    grad_input_offs = row_idx[:, None] * stride_gim + col_idx[None, :] * stride_gin
    tl.store(grad_input_ptr + grad_input_offs, grad_input, mask=mask)

class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        if not input.is_contiguous():
            input = input.contiguous()
        
        # Calculate M and N based on the dimension
        sizes = list(input.shape)
        M = 1
        for i in range(dim):
            M *= sizes[i]
        N = sizes[dim]
        for i in range(dim + 1, len(sizes)):
            M *= sizes[i]
            
        # Prepare output tensor
        output = torch.empty_like(input)
        
        # Calculate grid size
        BLOCK_M = 32
        BLOCK_N = 32
        grid = (triton.cdiv(M, BLOCK_M),)
        
        # Launch kernel
        log_softmax_kernel[grid](
            output, input, M, N,
            output.stride(0), 1,
            input.stride(0), 1,
            BLOCK_M, BLOCK_N
        )
        
        ctx.save_for_backward(output)
        ctx.dim = dim
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        dim = ctx.dim
        
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
            
        # Calculate M and N
        sizes = list(output.shape)
        M = 1
        for i in range(dim):
            M *= sizes[i]
        N = sizes[dim]
        for i in range(dim + 1, len(sizes)):
            M *= sizes[i]
            
        # Prepare gradient tensor
        grad_input = torch.empty_like(grad_output)
        
        # Calculate grid size
        BLOCK_M = 32
        BLOCK_N = 32
        grid = (triton.cdiv(M, BLOCK_M),)
        
        # Launch backward kernel
        log_softmax_backward_kernel[grid](
            grad_input, grad_output, output, M, N,
            grad_input.stride(0), 1,
            grad_output.stride(0), 1,
            output.stride(0), 1,
            BLOCK_M, BLOCK_N
        )
        
        return grad_input, None

def log_softmax(input, dim=-1):
    """
    Applies log softmax over a specified dimension.
    
    Args:
        input (torch.Tensor): input tensor
        dim (int): dimension to apply log softmax over
    
    Returns:
        torch.Tensor: output tensor of same shape as input
    """
    return LogSoftmax.apply(input, dim)
