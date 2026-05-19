import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(
    input_ptr,    # Pointer to input tensor
    other_ptr,    # Pointer to other tensor
    output_ptr,   # Pointer to output tensor
    alpha,        # Alpha scaling factor
    n_elements,   # Number of elements
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    # Calculate the program ID
    pid = tl.program_id(axis=0)
    # Calculate the block start
    block_start = pid * BLOCK_SIZE
    # Calculate offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    # Load other values
    y = tl.load(other_ptr + offsets, mask=mask)
    
    # Compute output: input + alpha * other
    output = x + alpha * y
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def add(input, other, *, alpha=1, out=None):
    """
    Adds a tensor or number scaled by alpha to the input tensor.
    
    Args:
        input (Tensor): the input tensor
        other (Tensor or Number): the tensor or number to add to input
        alpha (Number, optional): the multiplier for other (default: 1)
        out (Tensor, optional): the output tensor (default: None)
    
    Returns:
        Tensor: The result of input + alpha * other
    """
    # Handle scalar other
    if not isinstance(other, torch.Tensor):
        other = torch.full_like(input, other)
    
    # Ensure inputs are compatible
    if input.device != other.device:
        raise ValueError("Expected all tensors to be on the same device")
    
    # Handle broadcasting
    input, other = torch.broadcast_tensors(input, other)
    
    # Initialize output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    elif out.shape != input.shape:
        raise ValueError("Expected out tensor to have same shape as input")
    
    # Get total number of elements
    n_elements = input.numel()
    
    # Calculate grid and block sizes
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    add_kernel[grid](
        input_ptr=input.data_ptr(),
        other_ptr=other.data_ptr(),
        output_ptr=out.data_ptr(),
        alpha=alpha,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
