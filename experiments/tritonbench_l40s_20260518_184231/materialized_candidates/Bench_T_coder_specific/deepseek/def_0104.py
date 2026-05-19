import torch
from typing import Tuple

def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    out_deg = torch.rad2deg(input)
    out_sqrt = torch.sqrt(input)
    return out_deg, out_sqrt
