import torch
import triton
import triton.language as tl

@triton.jit
def _triton_sqrt(x):
    # Compute the square root of x
    return tl.sqrt(x)

def sqrt(input, *, out=None):
    # Check if input is a Tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    # Check if out is a Tensor
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    # Get input shape
    input_shape = input.shape
    # Reshape input to 1D
    input_reshaped = input.reshape(-1)
    # Get input stride
    input_stride = input_reshaped.stride(-1)
    # Get input dtype
    input_dtype = input_reshaped.dtype
    # If out is None, create a new tensor for output
    if out is None:
        out = torch.empty_like(input_reshaped, dtype=input_dtype)
    # Get output shape
    out_shape = out.shape
    # Reshape out to 1D
    out_reshaped = out.reshape(-1)
    # Get out stride
    out_stride = out_reshaped.stride(-1)
    # Get out dtype
    out_dtype = out_reshaped.dtype
    # Check if input and out have the same shape
    if input_shape != out_shape:
        raise ValueError("Shape of out and input must be the same")
    # Get the number of elements
    num_elements = 1
    for size in input_shape:
        num_elements *= size
    # Calculate grid size for triton kernel
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    # Call the triton kernel
    _triton_sqrt_kernel[grid](input_reshaped, out_reshaped, num_elements, input_stride, out_stride)
    # Reshape out to the original shape
    out = out.reshape(input_shape)
    return out
