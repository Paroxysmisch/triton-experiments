import torch
from torch import Tensor
import torch.nn.functional as F

def softmax_log(input: Tensor, dim: int = -1, dtype: torch.dtype = None) -> Tensor:
    if dtype is not None:
        input = input.to(dtype)
    return F.softmax(input.log(torch.get_default_dtype()), dim=dim)
