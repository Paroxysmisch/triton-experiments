import triton
import triton.language as tl
import torch

@triton.jit
def fused_recurrent_fwd_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, beta, scale,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    # Load q, k, v
    q = tl.load(q_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    k = tl.load(k_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    v = tl.load(v_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    
    # Element-wise operations
    product = q * k
    weighted_sum = product * v * beta * scale
    
    # Write to output
    tl.store(out_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), weighted_sum)

@triton.jit
def fused_recurrent_bwd_kernel(
    q_ptr, k_ptr, v_ptr, grad_out_ptr, grad_q_ptr, grad_k_ptr, grad_v_ptr, beta, scale,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    # Load inputs and gradients
    q = tl.load(q_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    k = tl.load(k_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    v = tl.load(v_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    grad_out = tl.load(grad_out_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    
    # Compute gradients
    grad_q = grad_out * k * v * beta * scale
    grad_k = grad_out * q * v * beta * scale
    grad_v = grad_out * q * k * beta * scale
    
    # Write gradients to output
    tl.store(grad_q_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), grad_q)
    tl.store(grad_k_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), grad_k)
    tl.store(grad_v_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), grad_v)

class FusedRecurrentFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, scale):
        BLOCK_SIZE = 128  # Example block size
        out = torch.empty_like(q)
        
        # Launch forward kernel
        grid = (q.numel() // BLOCK_SIZE,)
        fused_recurrent_fwd_kernel[grid](q, k, v, out, beta, scale, BLOCK_SIZE=BLOCK_SIZE)
        
        # Save for backward
        ctx.save_for_backward(q, k, v, torch.tensor(beta), torch.tensor(scale))
        
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, beta, scale = ctx.saved_tensors
        BLOCK_SIZE = 128  # Example block size
        
        # Prepare gradients
        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        
        # Launch backward kernel
        grid = (q.numel() // BLOCK_SIZE,)
        fused_recurrent_bwd_kernel[grid](
            q, k, v, grad_out, grad_q, grad_k, grad_v, beta.item(), scale.item(), BLOCK_SIZE=BLOCK_SIZE
        )
        
        return grad_q, grad_k, grad_v, None, None

def fused_recurrent_delta_rule(q, k, v, beta=1.0, scale=1.0):
    if not all(isinstance(t, torch.Tensor) for t in [q, k, v]):
        raise ValueError("All inputs must be PyTorch tensors")
    
    return FusedRecurrentFunction.apply(q, k, v, beta, scale)

q = torch.randn(256, device='cuda')
k = torch.randn(256, device='cuda')
v = torch.randn(256, device='cuda')

output = fused_recurrent_delta_rule(q, k, v, beta=0.9, scale=0.1)
