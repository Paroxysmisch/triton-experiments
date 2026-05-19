import torch
from torch import nn, Tensor

class ELULinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, alpha: float = 1.0):
        super(ELULinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.alpha = alpha

    def forward(self, input: Tensor) -> Tensor:
        return torch.nn.functional.elu(self.linear(input), alpha=self.alpha)
