import torch
from typing import Tuple

def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute the complementary error function (erfc)
    erfc_result = torch.erfc(input)
    
    # Compute the square root of each element in the input tensor
    sqrt_result = sqrt(input)
    
    # Return a tuple containing the erfc result and the square root result
    return erfc_result, sqrt_result
