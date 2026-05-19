import torch
import triton
import triton.language as tl

class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    @triton.jit
    def forward(ctx, Q, Q_row_stride, cos, sin, cos_row_stride, sin_row_stride, seqlen, head_dim, n_heads, BACKWARD_PASS, BLOCK_SIZE):
        Q = Q.contiguous()
        cos = cos.contiguous()
        sin = sin.contiguous()

        output = Q.new_empty_like(Q)
        grid = (triton.cdiv(Q.shape[0], BLOCK_SIZE), 1)
        _rope_embedding[grid](Q, Q_row_stride, cos, sin, cos_row_stride, sin_row_stride, seqlen, head_dim, n_heads, BACKWARD_PASS, BLOCK_SIZE)
        ctx.save_for_backward(Q, cos, sin)
        ctx.head_dim = head_dim
        return output

    @staticmethod
    @triton.jit
    def backward(ctx, grad_output):
        Q, cos, sin = ctx.saved_tensors
        head_dim = ctx.head_dim
        Q_row_stride = Q.stride(0)
        cos_row_stride = cos.stride(0)
        sin_row_stride = sin.stride(0)
        seqlen = Q.shape[0]
        n_heads = Q.shape[1] // head_dim
        BACKWARD_PASS = 1

        grad_Q = Q.new_empty_like(Q)
        grid = (triton.cdiv(Q.shape[0], BLOCK_SIZE), 1)
        _rope_embedding[grid](grad_Q, Q_row_stride, cos, sin, cos_row_stride, sin_row_stride, seqlen, head_dim, n_heads, BACKWARD_PASS, BLOCK_SIZE)
        return grad_Q, None, None, None, None, None, None, None, None, None

def fast_rope_embedding(Q, K, cos, sin, cos_row_stride, sin_row_stride, seqlen, head_dim, n_heads, BACKWARD_PASS, BLOCK_SIZE):
    Q = Q.transpose(0, 1)
    K = K.transpose(0, 1)
    Q = Fast_RoPE_Embedding.apply(Q, Q.stride(0), cos, sin, cos_row_stride, sin_row_stride, seqlen, head_dim, n_heads, BACKWARD_PASS, BLOCK_SIZE)
    K = Fast_RoPE_Embedding.apply(K, K.stride(0), cos, sin, cos_row_stride, sin_row_stride, seqlen, head_dim, n_heads, BACKWARD_PASS, BLOCK_SIZE)
    return Q.transpose(0, 1), K.transpose(0, 1)
