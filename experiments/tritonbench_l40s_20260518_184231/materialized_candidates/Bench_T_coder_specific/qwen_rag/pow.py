import triton
import triton.language as tl
import torch

@triton.jit
def pow_kernel(out_ptr, in_ptr, exp_ptr, n_elements, EXP_TYPE: tl.constexpr):
    # Get the global id of the current program/thread
    idx = tl.program_id(0)
    # Compute the stride for accessing elements in the input and output tensors
    stride = tl.cdiv(n_elements, tl.num_warps() * 32)
    # Load input and exponent values
    in_val = tl.load(in_ptr + idx * stride)
    exp_val = tl.load(exp_ptr + idx * stride)
    # Apply the power operation
    out_val = in_val ** exp_val
    # Store the result back to the output tensor
    tl.store(out_ptr + idx * stride, out_val)

def pow(input_tensor, exponent_tensor, out=None):
    # Check if output tensor is provided, otherwise create one
    if out is None:
        out = torch.empty_like(input_tensor)
    
    # Ensure that input and exponent tensors are on the same device
    assert input_tensor.device == exponent_tensor.device
    
    # Check if exponent is a scalar or tensor
    if isinstance(exponent_tensor, float):
        # If scalar, convert it to a tensor with the same shape as input
        exponent_tensor = torch.full_like(input_tensor, exponent_tensor)
    
    # Get the number of elements in the input tensor
    n_elements = input_tensor.numel()
    
    # Determine the block size for Triton kernel
    BLOCK_SIZE = 256
    grid_size = (tl.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the Triton kernel
    pow_kernel[grid_size](out, input_tensor, exponent_tensor, n_elements, EXP_TYPE=float)
    
    return out
