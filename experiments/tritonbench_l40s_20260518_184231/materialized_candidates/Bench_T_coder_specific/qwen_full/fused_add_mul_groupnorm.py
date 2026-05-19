import torch
import triton
import triton.language as tl
from torch.autograd.function import Function

class FusedAddMulGroupNorm(Function):
    @staticmethod
    def forward(ctx, x, y, weight, bias, num_groups, eps=1e-5, out=None):
        x = x.contiguous()
        y = y.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()

        if out is None:
            out = torch.empty_like(x)
        else:
            out = out.contiguous()

        x_dtype = x.dtype
        y_dtype = y.dtype
        weight_dtype = weight.dtype
        bias_dtype = bias.dtype
        out_dtype = out.dtype

        if x_dtype != out_dtype:
            if x_dtype.is_fp64():
                out = out.to(tl.float32)
            else:
                out = out.to(x_dtype)
        if y_dtype != out_dtype:
            if y_dtype.is_fp64():
                out.to(tl.float32)
            else:
                out.to(y_dtype)
        if weight_dtype != out_dtype:
            if weight_dtype.is_fp64():
                weight = weight.to(tl.float32)
            else:
                weight = weight.to(out_dtype)
        if bias_dtype != out_dtype:
            if bias_dtype.is_fp64():
                bias = bias.to(tl.float32)
            else:
                bias = bias.to(out_dtype)

        if x.dtype == torch.float16:
            x = x.to(tl.bfloat16)
            y = y.to(tl.bfloat16)

        if triton.__version__ >= "2.1.0":
            lafgng = _fused_add_mul_group_norm_forward_triton_ if x.is_contiguous() and y.is_contiguous() and out.is_contiguous() and weight.is_contiguous() and bias.is_contiguous() else _fused_add_mul_group_norm_forward_triton
        else:
            lafagng = _fused_add_mul_group_norm_forward_triton

        return lafagng(x, y, weight, bias, num_groups, out, eps)

def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None):
    return FusedAddMulGroupNorm.apply(input1, input2, weight, bias, num_groups, eps, out)
