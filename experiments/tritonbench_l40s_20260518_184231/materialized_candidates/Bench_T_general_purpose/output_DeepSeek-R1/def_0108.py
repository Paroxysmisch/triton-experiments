import torch
import torch.nn.functional as F

def grid_sample_with_affine(
    input: torch.Tensor,
    theta: torch.Tensor,
    size: torch.Size,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = False
) -> torch.Tensor:
    # Generate the affine grid
    grid = F.affine_grid(theta, size, align_corners=align_corners)
    # Apply grid sampling
    output = F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    return output
