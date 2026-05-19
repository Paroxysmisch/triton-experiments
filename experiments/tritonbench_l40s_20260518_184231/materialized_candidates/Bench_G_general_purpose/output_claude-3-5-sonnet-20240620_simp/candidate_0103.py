import torch
import triton
import triton.language as tl
import math

@triton.jit
def _rms_forward_kernel(
    x_ptr,          # pointer to input
    output_ptr,     # pointer to output
    weight_ptr,     # pointer to weight
    inv_rms_ptr,    # pointer to store inverse RMS for backward
    stride,         # stride for the batch dimension
    n_cols,         # number of columns
    BLOCK_SIZE: tl.constexpr,  # block size for parallelization
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute batch element to process
    offset = pid * stride
    
    # Initialize pointers
    x = tl.load(x_ptr + offset + tl.arange(0, BLOCK_SIZE))
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE))
    
    # Compute RMS
    square_sum = tl.sum(x * x, axis=0) / n_cols
    inv_rms = 1.0 / tl.sqrt(square_sum + 1e-6)
    
    # Store normalized output
    output = weight * x * inv_rms
    tl.store(output_ptr + offset + tl.arange(0, BLOCK_SIZE), output)
    
    # Store inv_rms for backward pass
    if pid == 0:
        tl.store(inv_rms_ptr, inv_rms)

@triton.jit
def _rms_backward_kernel(
    grad_output_ptr,    # pointer to gradient w.r.t output
    x_ptr,              # pointer to input
    weight_ptr,         # pointer to weight
    inv_rms_ptr,        # pointer to stored inverse RMS
    grad_input_ptr,     # pointer to gradient w.r.t input
    grad_weight_ptr,    # pointer to gradient w.r.t weight
    stride,             # stride for batch dimension
    n_cols,             # number of columns
    BLOCK_SIZE: tl.constexpr,  # block size for parallelization
):
    pid = tl.program_id(0)
    offset = pid * stride
    
    # Load saved values
    grad_output = tl.load(grad_output_ptr + offset + tl.arange(0, BLOCK_SIZE))
    x = tl.load(x_ptr + offset + tl.arange(0, BLOCK_SIZE))
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE))
    inv_rms = tl.load(inv_rms_ptr)
    
    # Compute gradients
    normalized_x = x * inv_rms
    grad_input = grad_output * weight * inv_rms
    grad_weight = tl.sum(grad_output * normalized_x, axis=0)
    
    # Store results
    tl.store(grad_input_ptr + offset + tl.arange(0, BLOCK_SIZE), grad_input)
    if pid == 0:
        tl.store(grad_weight_ptr + tl.arange(0, BLOCK_SIZE), grad_weight)

class FastRMSLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Determine dimensions and block size
        batch_size, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        
        # Allocate output tensors
        output = torch.empty_like(x)
        inv_rms = torch.empty(1, device=x.device, dtype=x.dtype)
        
        # Launch kernel
        grid = (batch_size,)
        _rms_forward_kernel[grid](
            x.data_ptr(),
            output.data_ptr(),
            weight.data_ptr(),
            inv_rms.data_ptr(),
            n_cols,
            n_cols,
            BLOCK_SIZE,
        )
        
        # Save for backward
        ctx.save_for_backward(x, weight, inv_rms)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, inv_rms = ctx.saved_tensors
        batch_size, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        
        # Allocate gradient tensors
        grad_input = torch.empty_like(x)
        grad_weight = torch.empty_like(weight)
        
        # Launch kernel
        grid = (batch_size,)
        _rms_backward_kernel[grid](
            grad_output.data_ptr(),
            x.data_ptr(),
            weight.data_ptr(),
            inv_rms.data_ptr(),
            grad_input.data_ptr(),
            grad_weight.data_ptr(),
            n_cols,
            n_cols,
            BLOCK_SIZE,
        )
        
        return grad_input, grad_weight

class RMSLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.fast_rms_ln = FastRMSLayerNorm.apply
        
    def forward(self, x):
        return self.fast_rms_ln(x, self.weight)

# Example usage
def test_rms_layernorm():
    batch_size = 32
    hidden_size = 512
    
    layer = RMSLayerNorm(hidden_size).cuda()
    x = torch.randn(batch_size, hidden_size, device='cuda', requires_grad=True)
    
    # Forward pass
    out = layer(x)
    
    # Backward pass
    loss = out.sum()
    loss.backward()
    
    print("Forward pass shape:", out.shape)
    print("Gradient shape:", x.grad.shape)

if __name__ == "__main__":
    test_rms_layernorm()
