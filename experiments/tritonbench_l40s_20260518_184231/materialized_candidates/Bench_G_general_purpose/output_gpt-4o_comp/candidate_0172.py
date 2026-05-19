import torch
import triton
import triton.language as tl

# Define constants for block sizes
BLOCK_M = 128
BLOCK_N = 128

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8),
    ],
    key=['M', 'N']
)
@triton.jit
def log_softmax_kernel(X_ptr, Y_ptr, M, N, stride_xm, stride_xn, stride_ym, stride_yn, **meta):
    BLOCK_M = meta['BLOCK_M']
    BLOCK_N = meta['BLOCK_N']
    
    # Define program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load data
    X = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=-float('inf'))
    
    # Compute row-wise max
    row_max = tl.max(X, axis=1)
    
    # Subtract max and exponentiate
    X_exp = tl.exp(X - row_max[:, None])
    
    # Compute sum of exponentials
    sum_exp = tl.sum(X_exp, axis=1)
    
    # Compute log softmax
    Y = X - row_max[:, None] - tl.log(sum_exp[:, None])
    
    # Store result
    tl.store(Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, Y, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def log_softmax_backward_kernel(grad_output_ptr, output_ptr, grad_input_ptr, M, N, stride_gom, stride_gon, stride_om, stride_on, stride_gim, stride_gin, **meta):
    BLOCK_M = meta['BLOCK_M']
    BLOCK_N = meta['BLOCK_N']
    
    # Define program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load data
    grad_output = tl.load(grad_output_ptr + offs_m[:, None] * stride_gom + offs_n[None, :] * stride_gon, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    output = tl.load(output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    
    # Compute sum of gradients
    sum_grad = tl.sum(grad_output, axis=1)
    
    # Compute gradient of input
    grad_input = grad_output - tl.exp(output) * sum_grad[:, None]
    
    # Store result
    tl.store(grad_input_ptr + offs_m[:, None] * stride_gim + offs_n[None, :] * stride_gin, grad_input, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        M, N = x.shape
        y = torch.empty_like(x)
        
        # Launch kernel
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        log_softmax_kernel[grid](x, y, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1))
        
        ctx.save_for_backward(y)
        ctx.dim = dim
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        M, N = grad_output.shape
        grad_input = torch.empty_like(grad_output)
        
        # Launch kernel
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        log_softmax_backward_kernel[grid](grad_output, y, grad_input, M, N, grad_output.stride(0), grad_output.stride(1), y.stride(0), y.stride(1), grad_input.stride(0), grad_input.stride(1))
        
        return grad_input, None

def log_softmax(x, dim=-1):
    if not x.is_contiguous():
        x = x.contiguous()
    return LogSoftmax.apply(x, dim)
