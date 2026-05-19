def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    # Normalize grid values
    grid = grid * 2 - 1
    # Handle NaN values
    grid[torch.isnan(grid)] = -1

    # Call the Triton function
    output = torch.ops.triton.grid_sample(input, grid, mode, padding_mode, align_corners)

    return output
