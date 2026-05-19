import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def mul_relu_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    inplace: tl.constexpr,
    is_scalar: tl.constexpr,
    scalar_value: tl.float32 = 0.0
):
    # Calculate the block and program IDs
    pid = tl.program_id(0)
    block_size = tl.num_programs(0)
    
    # Calculate the index of the first element this program will handle
    offsets = pid * block_size + tl.arange(0, block_size)
    
    # Mask to handle cases where offsets exceed n_elements
    mask = offsets < n_elements
    
    # Load input and other (either tensor or scalar)
    input_val = tl.load(input_ptr + offsets, mask=mask)
    if is_scalar:
        other_val = scalar_value
    else:
        other_val = tl.load(other_ptr + offsets, mask=mask)
    
    # Perform element-wise multiplication
    result = input_val * other_val
    
    # Apply ReLU (max with zero)
    result = tl.where(result > 0, result, 0.0)
    
    # Store the result, in-place if requested
    if inplace:
        tl.store(input_ptr + offsets, result, mask=mask)
    else:
        tl.store(output_ptr + offsets, result, mask=mask)

def mul_relu(input: Tensor, other, inplace=False, out=None) -> Tensor:
    """
    Performs element-wise multiplication of `input` and `other`, followed by ReLU.
    
    Args:
        input (Tensor): The input tensor to be multiplied.
        other (Tensor or Number): The tensor or number to multiply with `input`.
        inplace (bool, optional): If True, modifies `input` in-place, if possible. Default is False.
        out (Tensor, optional): The output tensor.

    Returns:
        Tensor: The resulting tensor after multiplication and ReLU.
    """
    # Ensure `other` is a tensor or a scalar
    if isinstance(other, Tensor):
        is_scalar = False
        other_ptr = other
        scalar_value = 0.0
    else:
        is_scalar = True
        other_ptr = None
        scalar_value = float(other)
    
    # Determine the output tensor
    if inplace:
        output = input
    else:
        if out is None:
            output = torch.empty_like(input)
        else:
            output = out
    
    # Get the number of elements
    n_elements = input.numel()
    
    # Launch the Triton kernel
    block_size = triton.next_power_of_2(min(1024, n_elements))
    grid = (n_elements + block_size - 1) // block_size
    
    mul_relu_kernel[grid](
        input,
        other_ptr,
        output,
        n_elements,
        inplace,
        is_scalar,
        scalar_value,
        num_warps=4
    )
    
    return output

# Example usage
input_tensor = torch.tensor([1.0, -2.0, 3.0, -4.0], device='cuda')
other_tensor = torch.tensor([2.0, 2.0, 2.0, 2.0], device='cuda')
result = mul_relu(input_tensor, other_tensor)
print(result)  # Output should be [2.0, 0.0, 6.0, 0.0]
