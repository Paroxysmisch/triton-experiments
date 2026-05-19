import torch
import triton
import triton.language as tl

class ChunkRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, initial_state=None, decay_func=None, NT=None, D=None):
        # Allocate buffers
        b_h = torch.zeros_like(q)
        b_o = torch.zeros_like(q)
        b_s = torch.zeros_like(q)
        
        # Launch kernel
        grid = (q.shape[0], (q.shape[1] + 255) // 256)
        block = (256, 1, 1)
        chunk_retention_fwd_kernel_h[grid, block](
            k, v, initial_state, b_h, decay_func, NT, D, USE_INITIAL_STATE=1, STORE_FINAL_STATE=1
        )
        chunk_retention_fwd_kernel_o[grid, block](
            q, k, v, b_h, decay_func, b_o, b_s, NT, D
        )
        
        ctx.save_for_backward(q, k, v, initial_state, decay_func, NT, D, b_h, b_o, b_s)
        return b_o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, initial_state, decay_func, NT, D, b_h, b_o, b_s = ctx.saved_tensors
        
        # Allocate buffers for gradients
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)
        
        # Launch backward kernel
        grid = (q.shape[0], (q.shape[1] + 255) // 256)
        block = (256, 1, 1)
        chunk_retention_bwd_kernel_dh[grid, block](
            grad_output, b_h, grad_q, NT, D
        )
        chunk_retention_bwd_kernel_dqkv[grid, block](
            q, k, v, grad_output, b_h, b_o, b_s, grad_q, grad_k, grad_v, NT, D
        )
        
        return grad_q, grad_k, grad_v, None, None, None, None

def chunk_retention(q, k, v, initial_state=None, decay_func=None, NT=None, D=None):
    return ChunkRetentionFunction.apply(q, k, v, initial_state, decay_func, NT, D)
