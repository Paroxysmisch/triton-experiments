import torch
from typing import List


def broadcast_tensors(*tensors) -> List[torch.Tensor]:
    return torch.broadcast_tensors(*tensors)
