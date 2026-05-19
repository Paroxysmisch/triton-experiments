import torch
from torch.autograd import Function

class FusedRecurrentFunction(Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, initial_state, output, final_state):
        B, H, T, K, V = q.shape
        BK = 32  # Block size for q
        BV = 32  # Block size for v
        
        # Allocate memory for output and final_state
        output = torch.zeros_like(q, device=q.device)
        final_state = torch.zeros_like(q[:, :, 0, :, :], device=q.device) if final_state is not None else None
        
        # Launch the forward kernel
        fused_recurrent_fwd_kernel[BK, BK, BK](q, k, v, beta, initial_state, output, final_state,
                                               B, H, T, K, V, BK, BV)
        
        # Save tensors for backward pass
        ctx.save_for_backward(q, k, v, beta, initial_state, output, final_state)
        
        return output, final_state
    
    @staticmethod
    def backward(ctx, grad_output, grad_final_state):
        q, k, v, beta, initial_state, output, final_state = ctx.saved_tensors
        
        B, H, T, K, V = q.shape
        BK = 32  # Block size for q
        BV = 32  # Block size for v
        
        # Allocate memory for gradients
        grad_q = torch.zeros_like(q, device=q.device)
        grad_k = torch.zeros_like(k, device=k.device)
        grad_v = torch.zeros_like(v, device=v.device)
        
        # Launch the backward kernel
        fused_recurrent_bwd_kernel[BK, BK, BK](q, k, v, beta, initial_state, output, final_state, grad_output, grad_q, grad_k, grad_v,
                                               B, H, T, K, V, BK, BV)
        
        return grad_q, grad_k, grad_v, None, None, grad_output, None
