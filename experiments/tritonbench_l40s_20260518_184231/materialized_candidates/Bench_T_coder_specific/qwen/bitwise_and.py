import triton
import torch

@torch.jit.script
def bitwise_and(input, other, out=None):
    # Check if the inputs are of supported types
    if input.dtype not in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool] or \
       other.dtype not in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool]:
        raise ValueError("Input tensors must be of integral or Boolean types")

    # Get the device type
    device_type = input.device.type

    # Create the output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Launch the Triton kernel
    n_elements = input.numel()
    grid_size = -1  # Let Triton automatically determine the grid size
    block_size = 1024  # Block size for parallel execution

    # Call the Triton kernel
    bitwise_and_kernel[grid_size, block_size](input.data_ptr(), other.data_ptr(), out.data_ptr(), n_elements)

    return out
