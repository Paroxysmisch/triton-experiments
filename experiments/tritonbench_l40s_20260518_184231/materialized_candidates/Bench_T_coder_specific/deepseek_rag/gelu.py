import torch
from torch.autograd import Function

class GELU(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return 0.5 * x * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        tmp = 2 / torch.sqrt(torch.tensor(3.14159265)) * torch.exp(-x.pow(2))
        return grad_output * (0.5 * x * tmp + 0.5 * torch.erf(x / torch.sqrt(torch.tensor(2.0))))
