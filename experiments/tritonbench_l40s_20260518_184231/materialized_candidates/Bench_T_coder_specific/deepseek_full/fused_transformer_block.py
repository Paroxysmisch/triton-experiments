import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def fused_transformer_block(
    input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, **meta
):
    z1 = tl.dot(input, weight1)
    z2 = tl.softmax(z1, axis=-1)
    z3 = tl.dropout(z2, dropout_p, 0)
    z4 = tl.dot(z3, weight2)
    y = tl.layer_norm(z4 + residual, eps, meta["z4_scale"], meta["z4_bias"])
    return y

class FusedTransformerBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight1, weight2, residual, dropout_p, eps):
        output_shape = x.shape[:-1] + weight2.shape[-1:]
        y = torch.empty(output_shape, device=x.device, dtype=x.dtype)
        meta = {"z4_scale": weight2.shape[-1] // 2}
        grid = lambda meta: (triton.cdiv(1, meta["BLOCK"]),)
        fused_transformer_block[grid](x, weight1, weight2, residual, dropout_p, eps, out=y, **meta)
        ctx.save_for_backward(x, weight1, weight2, residual)
        ctx.dropout_p = dropout_p
        ctx.training = x.requires_grad
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight1, weight2, residual = ctx.saved_tensors
        dropout_p = ctx.dropout_p
        eps = 1e-5
        output_shape = dy.shape[:-1] + weight1.shape[-1:]
        dx = torch.empty(output_shape, device=x.device, dtype=x.dtype)
        dweight1 = torch.empty_like(weight1)
        dweight2 = torch.empty_like(weight2)
        dresidual = torch.empty_like(residual)
        meta = {"z1_scale": weight1.shape[-1], "z4_scale": weight2.shape[-1] // 2}
        grid = lambda meta: (triton.cdiv(1, meta["BLOCK"]),)
        fused_transformer_block[grid](dy, weight2, weight1, residual, dropout_p, eps, out=dx, **meta)
        return dx, dweight1, dweight2, dresidual, None, None

def fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, out=None):
    return FusedTransformerBlock.apply(input, weight1, weight2, residual, dropout_p, eps)
