import torch
from torch.autograd import Function
from .fused_add_mul_groupnorm import fused_add_mul_groupnorm_cuda

class FusedAddMulGroupNorm(Function):
    @staticmethod
    def forward(ctx, input1, input2, weight, bias, num_groups, eps=1e-5):
        ctx.save_for_backward(input1, input2, weight, bias, num_groups, eps)
        output = torch.empty_like(input1)
        fused_add_mul_groupnorm_cuda(input1, input2, weight, bias, output, num_groups, eps)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2, weight, bias, num_groups, eps = ctx.saved_tensors
        grad_input1 = torch.zeros_like(input1)
        grad_input2 = torch.zeros_like(input2)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)

        fused_add_mul_groupnorm_cuda(grad_output, input2, weight, bias, grad_input1, num_groups, eps)
        fused_add_mul_groupnorm_cuda(input1, grad_output, weight, bias, grad_input2, num_groups, eps)
        fused_add_mul_groupnorm_cuda(input1, input2, grad_output, bias, grad_weight, num_groups, eps)
        fused_add_mul_groupnorm_cuda(input1, input2, weight, grad_output, grad_bias, num_groups, eps)

        return grad_input1, grad_input2, grad_weight, grad_bias, None, None

def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, out=None):
    return FusedAddMulGroupNorm.apply(input1, input2, weight, bias, num_groups, eps)
