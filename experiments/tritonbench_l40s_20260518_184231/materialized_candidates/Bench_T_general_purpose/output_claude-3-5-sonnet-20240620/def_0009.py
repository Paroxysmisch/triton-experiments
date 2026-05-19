{{ code }}
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def grid_sample_kernel(input_ptr, grid_ptr, output_ptr, 
                       mode, padding_mode, align_corners, 
                       input_shape, grid_shape, output_shape):
    # ... existing code ...
    # Implement nearest or bilinear interpolation based on mode
    if mode == 'bilinear':
        # ... bilinear interpolation logic ...
    elif mode == 'nearest':
        # ... nearest interpolation logic ...
    # Handle padding_mode and NaN values
    # ... existing code ...

def grid_sample(input: Tensor, grid: Tensor, mode: str = 'bilinear', 
                padding_mode: str = 'zeros', align_corners: bool = False) -> Tensor:
    # Validate input shapes and types
    # ... existing code ...
    
    # Prepare Triton kernel launch parameters
    input_shape = input.shape
    grid_shape = grid.shape
    output_shape = ...  # Calculate output shape based on input and grid
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Launch the Triton kernel
    grid_sample_kernel[(grid_shape[0], ...)](input.data_ptr(), grid.data_ptr(), 
                                              output.data_ptr(), 
                                              mode, padding_mode, align_corners, 
                                              input_shape, grid_shape, output_shape)
    
    return output
