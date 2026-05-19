import triton
import triton.language as tl

@triton.jit
def fused_recurrent_hgrn_fwd_kernel(x_ptr, g_ptr, h0_ptr, o_ptr, T, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load initial state
    h_t = tl.load(h0_ptr + offset, mask=offset < N, other=0.0)
    
    for t in range(T):
        # Compute the index for time step t
        x_idx = t * N + offset
        g_idx = t * N + offset
        
        # Load x_t and g_t
        x_t = tl.load(x_ptr + x_idx, mask=offset < N, other=0.0)
        g_t = tl.load(g_ptr + g_idx, mask=offset < N, other=0.0)
        
        # Compute h_t and o_t
        h_t = g_t * h_t + x_t
        tl.store(o_ptr + x_idx, h_t, mask=offset < N)

#### Backward Kernel


import torch
from torch.autograd import Function

class FusedRecurrentHGRNFunction(Function):
    @staticmethod
    def forward(ctx, x, g, h0=None):
        # Allocate output tensor
        o = torch.empty_like(x)
        
        # Get dimensions
        T, N = x.shape
        
        # Launch forward kernel
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        fused_recurrent_hgrn_fwd_kernel[grid](x, g, h0, o, T, N, BLOCK_SIZE=1024)
        
        # Save tensors for backward
        ctx.save_for_backward(x, g, h0, o)
        return o

    @staticmethod
    def backward(ctx, grad_o):
        x, g, h0, o = ctx.saved_tensors
        
        # Allocate gradient tensors
        grad_x = torch.empty_like(x)
        grad_g = torch.empty_like(g)
        
        # Get dimensions
        T, N = x.shape
        
        # Launch backward kernel
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        fused_recurrent_hgrn_bwd_kernel[grid](grad_o, x, g, grad_x, grad_g, h0, T, N, BLOCK_SIZE=1024)
        
        return grad_x, grad_g, None

def fused_recurrent_hgrn(x, g, initial_state=None):
    return FusedRecurrentHGRNFunction.apply(x, g, initial_state)

### Usage

You can now use `fused_recurrent_hgrn` in your PyTorch models. Here is an example:
