import torch
from typing import Tuple

def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    erfc_out = torch.special.erfc(input)
    sqrt_out = torch.sqrt(input)
    return erfc_out, sqrt_out
