import torch
import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(
    output_ptr, input_ptr, M, N, K,
    stride_om, stride_on, stride_ok,
    stride_im, stride_in, stride_ik,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Position of elements processed by this program
    pid = tl.program_id(0)
    
    # Compute offsets for this block
    block_start_m = pid * BLOCK_M
    block_start_n = tl.program_id(1) * BLOCK_N
    
    # Create offsets for this program
    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_N)
    
    # Create mask to handle partial blocks
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Compute input offsets
    input_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    
    # Load input data
    x = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    x_max = tl.max(x, axis=1)[:, None]
    
    # Compute exponentials and sum
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=1)[:, None]
    
    # Compute log softmax
    x_log_softmax = x - x_max - tl.log(x_sum)
    
    # Store results
    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(output_ptrs, x_log_softmax, mask=mask)

@triton.jit
def log_softmax_backward_kernel(
    grad_input_ptr, grad_output_ptr, output_ptr, M, N, K,
    stride_gim, stride_gin, stride_gik,
    stride_gom, stride_gon, stride_gok,
    stride_om, stride_on, stride_ok,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Similar block computation as forward
    block_start_m = pid * BLOCK_M
    block_start_n = tl.program_id(1) * BLOCK_N
    
    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_N)
    
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Load gradients and outputs
    grad_output_ptrs = grad_output_ptr + offs_m[:, None] * stride_gom + offs_n[None, :] * stride_gon
    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    
    grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0)
    output = tl.load(output_ptrs, mask=mask, other=0.0)
    
    # Compute gradient
    softmax = tl.exp(output)
    sum_grad_output = tl.sum(grad_output * softmax, axis=1)[:, None]
    grad_input = softmax * (grad_output - sum_grad_output)
    
    # Store gradients
    grad_input_ptrs = grad_input_ptr + offs_m[:, None] * stride_gim + offs_n[None, :] * stride_gin
    tl.store(grad_input_ptrs, grad_input, mask=mask)

class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=-1):
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get dimensions
        shape = x.shape
        dim = dim if dim >= 0 else len(shape) + dim
        M = shape[dim]
        N = 1
        for i in range(len(shape)):
            if i != dim:
                N *= shape[i]
        
        # Allocate output
        output = torch.empty_like(x)
        
        # Configure grid and block sizes
        BLOCK_M = 32
        BLOCK_N = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        
        # Launch kernel
        log_softmax_kernel[grid](
            output, x,
            M, N, 1,  # K is not used in this case
            *output.stride(),
            *x.stride(),
            BLOCK_M, BLOCK_N,
        )
        
        ctx.save_for_backward(output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        dim = ctx.dim
        
        # Ensure grad_output is contiguous
        grad_output = grad_output.contiguous()
        
        # Get dimensions
        shape = output.shape
        M = shape[dim]
        N = 1
        for i in range(len(shape)):
            if i != dim:
                N *= shape[i]
        
        # Allocate gradient input
        grad_input = torch.empty_like(grad_output)
        
        # Configure grid and block sizes
        BLOCK_M = 32
        BLOCK_N = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        
        # Launch backward kernel
        log_softmax_backward_kernel[grid](
            grad_input, grad_output, output,
            M, N, 1,
            *grad_input.stride(),
            *grad_output.stride(),
            *output.stride(),
            BLOCK_M, BLOCK_N,
        )
        
        return grad_input, None

def log_softmax(x, dim=-1):
    """
    Applies log softmax over a dimension of the input tensor.
    
    Args:
        x (torch.Tensor): input tensor
        dim (int): dimension to compute log softmax over
    
    Returns:
        torch.Tensor: output tensor with same shape as input
    """
    return LogSoftmax.apply(x, dim)
