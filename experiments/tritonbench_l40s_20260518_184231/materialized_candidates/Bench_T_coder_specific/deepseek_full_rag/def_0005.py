import triton
import torch

@triton.jit
def relu_sqrt_kernel(input, out, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create mask for boundary conditions
    mask = offset < n_elements
    # Load elements from input tensor with mask
    input_value = tl.load(input + offset, mask=mask)
    # Apply ReLU function to input
    relu_value = tl.maximum(input_value, 0.0)
    # Compute square root of the result
    sqrt_value = tl.sqrt(relu_value)
    # Store the result in output tensor
    tl.store(out + offset, sqrt_value, mask=mask)

def relu_sqrt(input, inplace=False, out=None) -> torch.Tensor:
    # Check if output tensor is provided
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a Tensor")
        if out.device != input.device:
            raise ValueError("out must be on the same device as input")
    else:
        # Create an output tensor if not provided
        out = torch.empty_like(input)
    
    # Check if inplace is True
    if inplace:
        if input.device != out.device:
            raise ValueError("input and out must be on the same device")
        if input.dtype != torch.float32:
            raise ValueError("input must be a float32 tensor")
        if input.stride(0) == 1 and out.stride(0) == 1:
            # Apply ReLU and compute square root in-place
            relu_sqrt_kernel[(ceildiv(len(input), 64),)](input, input, len(input), 64)
            return input
    else:
        # Apply ReLU and compute square root
        relu_sqrt_kernel[(ceildiv(len(input), 64),)](input, out, len(input), 64)
        return out
