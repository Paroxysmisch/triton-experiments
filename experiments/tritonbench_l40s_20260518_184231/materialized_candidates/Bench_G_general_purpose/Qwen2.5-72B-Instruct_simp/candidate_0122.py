import torch
import triton
import triton.language as tl

class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        # Save the input tensors for the backward pass
        ctx.save_for_backward(Q, cos, sin)
        
        # Launch the Triton kernel
        grid = (Q.numel() // 2, )
        _rope_embedding[grid](Q, cos, sin, 
                              Q.stride(0), Q.stride(1), 
                              cos.stride(0), cos.stride(1), 
                              sin.stride(0), sin.stride(1), 
                              Q.size(1), Q.size(0))
        
        return Q

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved tensors
        Q, cos, sin = ctx.saved_tensors
        
        # Launch the Triton kernel for the backward pass
        grid = (Q.numel() // 2, )
        _rope_embedding[grid](grad_output, cos, sin, 
                              grad_output.stride(0), grad_output.stride(1), 
                              cos.stride(0), cos.stride(1), 
                              sin.stride(0), sin.stride(1), 
                              Q.size(1), Q.size(0))
        
        return grad_output, None, None

def fast_rope_embedding(Q, K, cos, sin):
    # Apply RoPE to both query and key embeddings
    Q = Fast_RoPE_Embedding.apply(Q, cos, sin)
    K = Fast_RoPE_Embedding.apply(K, cos, sin)
    return Q, K
