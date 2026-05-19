import torch

def grid_sample_wrapper(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    # Check if grid values are outside the [-1, 1] range
    mask = (grid < -1) | (grid > 1)

    # Replace grid values outside the [-1, 1] range with zeros
    grid[mask] = 0

    # Call the original PyTorch grid_sample function
    output = torch.nn.functional.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    return output
