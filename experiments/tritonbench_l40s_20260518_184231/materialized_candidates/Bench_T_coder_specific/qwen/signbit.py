import torch
import triton
import triton.language as tl

# Define the Triton kernel
signbit_kernel = triton.compile(signbit_kernel)

def signbit(input_tensor, out=None):
    """
    Tests if each element of the input tensor has its sign bit set or not.
    Handles signed zeros, so negative zero (-0) returns True.
    
    Args:
        input_tensor (torch.Tensor): The input tensor.
        out (torch.Tensor, optional): The output tensor.
    
    Returns:
        torch.Tensor: The output tensor containing boolean values indicating whether the sign bit is set.
    """
    # Get the device and dtype of the input tensor
    device = input_tensor.device
    dtype = input_tensor.dtype
    
    # Check if the output tensor is provided
    if out is None:
        out = torch.empty_like(input_tensor, dtype=torch.bool)
    
    # Launch the Triton kernel
    n_elements = input_tensor.numel()
    grid_size = (n_elements + 255) // 256
    signbit_kernel[grid_size, 1](input_tensor.data_ptr(), out.data_ptr(), n_elements)
    
    return out
