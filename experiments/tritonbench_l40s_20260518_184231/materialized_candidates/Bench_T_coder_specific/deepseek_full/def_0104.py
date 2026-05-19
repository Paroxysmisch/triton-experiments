import torch
from typing import Tuple

def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert radians to degrees and calculate the square root for each element in the input tensor.

    Args:
        input (Tensor): The input tensor with angles in radians.

    Returns:
        Tuple[Tensor, Tensor]: A tuple where the first element is the converted degrees
            and the second is the square root of the input tensor elements.
    """
    degrees = torch.rad2deg(input)
    sqrt = torch.sqrt(input)
    return degrees, sqrt
