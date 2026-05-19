import torch
import triton
import triton.language as tl

# Define constants
BLOCK_SIZE = 128  # Example block size
NUM_WARPS = 4     # Example number of warps

@triton.jit
def _rms_layernorm_forward(input_ptr, weight_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    offsets = row_idx * N + tl.arange(0, BLOCK_SIZE)
    x = tl.load(input_ptr + offsets, mask=offsets < N)
    
    # Compute mean square
    mean_square = tl.sum(x * x, axis=0) / N
    inv_sqrt_var = tl.rsqrt(mean_square)
    
    # Normalize and scale
    y = x * inv_sqrt_var
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N)
    y = y * weight
    
    # Store result
    tl.store(output_ptr + offsets, y, mask=offsets < N)

@triton.jit
def _rms_layernorm_backward(grad_output_ptr, input_ptr, weight_ptr, grad_input_ptr, grad_weight_ptr, N, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    offsets = row_idx * N + tl.arange(0, BLOCK_SIZE)
    grad_output = tl.load(grad_output_ptr + offsets, mask=offsets < N)
    input = tl.load(input_ptr + offsets, mask=offsets < N)
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N)
    
    # Compute mean square
    mean_square = tl.sum(input * input, axis=0) / N
    inv_sqrt_var = tl.rsqrt(mean_square)
    
    # Compute gradients
    grad_input = grad_output * inv_sqrt_var * weight
    grad_weight = tl.sum(grad_output * input * inv_sqrt_var, axis=0)
    
    # Store gradients
    tl.store(grad_input_ptr + offsets, grad_input, mask=offsets < N)
    tl.atomic_add(grad_weight_ptr + tl.arange(0, BLOCK_SIZE), grad_weight, mask=tl.arange(0, BLOCK_SIZE) < N)

@triton.jit
def _gemma_rms_layernorm_forward(input_ptr, weight_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    offsets = row_idx * N + tl.arange(0, BLOCK_SIZE)
    x = tl.load(input_ptr + offsets, mask=offsets < N)
    
    # Compute mean square
    mean_square = tl.sum(x * x, axis=0) / N
    inv_sqrt_var = tl.rsqrt(mean_square)
    
    # Normalize and scale
    y = x * inv_sqrt_var
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N)
    y = y * (weight + 1.0)  # Additional constant 1.0 added
    
    # Store result
    tl.store(output_ptr + offsets, y, mask=offsets < N)

class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        N = input.shape[1]
        output = torch.empty_like(input)
        
        grid = (input.shape[0],)
        _rms_layernorm_forward[grid](input, weight, output, N, BLOCK_SIZE=BLOCK_SIZE)
        
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        N = input.shape[1]
        
        grad_input = torch.empty_like(input)
        grad_weight = torch.zeros_like(weight)
        
        grid = (input.shape[0],)
        _rms_layernorm_backward[grid](grad_output, input, weight, grad_input, grad_weight, N, BLOCK_SIZE=BLOCK_SIZE)
        
        return grad_input, grad_weight

def fast_rms_layernorm(input, weight):
    return Fast_RMS_Layernorm.apply(input, weight)

# Example usage
input = torch.randn(32, 128, device='cuda')
weight = torch.randn(128, device='cuda')
output = fast_rms_layernorm(input, weight)
