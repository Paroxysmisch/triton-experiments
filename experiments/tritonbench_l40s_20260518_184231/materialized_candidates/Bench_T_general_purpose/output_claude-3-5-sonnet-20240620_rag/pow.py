import triton
import triton.language as tl
import torch

@triton.jit
def pow_kernel(output_ptr, input_ptr, exponent_ptr, input_row_stride, output_row_stride, n_elements, is_scalar: tl.constexpr):
    # Get the index for the current program/thread
    idx = tl.program_id(0)
    
    # Load the input element
    x = tl.load(input_ptr + idx)
    
    # Determine the exponent value
    if is_scalar:
        exponent = tl.load(exponent_ptr)  # Load scalar exponent
    else:
        exponent = tl.load(exponent_ptr + idx)  # Load tensor exponent
    
    # Compute the power operation
    result = x ** exponent
    
    # Store the result back to the output tensor
    tl.store(output_ptr + idx, result)

def pow(input: torch.Tensor, exponent: float or torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Determine the number of elements
    n_elements = input.numel()
    
    # Create an output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure that the input and output tensors are on CUDA device
    assert input.is_cuda and out.is_cuda
    
    # Check if the exponent is a scalar or tensor
    is_scalar = isinstance(exponent, float)
    
    # Create a tensor for scalar exponent if needed
    if is_scalar:
        exponent_tensor = torch.tensor(exponent, device=input.device)
    else:
        exponent_tensor = exponent
    
    # Launch the Triton kernel
    pow_kernel[(n_elements, )](out, input, exponent_tensor, input.stride(0), out.stride(0), n_elements, is_scalar=is_scalar)
    
    return out
