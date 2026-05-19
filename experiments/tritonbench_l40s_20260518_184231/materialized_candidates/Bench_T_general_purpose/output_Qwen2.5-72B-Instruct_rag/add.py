import triton
import triton.language as tl
import torch

# Triton kernel for adding a tensor or number to another tensor
@triton.jit
def add_kernel(
    input_ptr, other_ptr, output_ptr, alpha, 
    input_shape, other_shape, output_shape,
    input_strides, other_strides, output_strides,
    input_numel, other_numel, output_numel,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_numel

    # Calculate the linear indices for input and other based on broadcasting
    input_indices = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    other_indices = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for dim in range(len(output_shape)):
        input_indices += (offsets // output_strides[dim]) % input_shape[dim] * input_strides[dim]
        other_indices += (offsets // output_strides[dim]) % other_shape[dim] * other_strides[dim]

    # Load the data
    input_data = tl.load(input_ptr + input_indices, mask=mask)
    other_data = tl.load(other_ptr + other_indices, mask=mask)

    # Perform the addition
    result = input_data + alpha * other_data

    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)

def add(input, other, *, alpha=1, out=None):
    """
    Adds the tensor or number 'other', scaled by 'alpha', to the 'input' tensor.
    Supports broadcasting to a common shape, type promotion, and accepts integer, float, and complex inputs.
    
    Parameters:
    - input (Tensor): The input tensor.
    - other (Tensor or Number): The tensor or number to add to input.
    - alpha (Number): The multiplier for other.
    - out (Tensor, optional): The output tensor.
    
    Returns:
    - Tensor: The resulting tensor after performing the addition.
    """
    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input, dtype=input.dtype, device=input.device)
    else:
        assert out.shape == input.shape, "Output tensor shape must match input tensor shape"

    # Convert alpha to the appropriate tensor type
    alpha = torch.tensor(alpha, dtype=input.dtype, device=input.device)

    # Determine the shapes and strides for broadcasting
    input_shape = input.shape
    other_shape = other.shape if isinstance(other, torch.Tensor) else (1,)
    output_shape = torch.broadcast_shapes(input_shape, other_shape)

    input_strides = input.stride()
    other_strides = other.stride() if isinstance(other, torch.Tensor) else (0,)
    output_strides = out.stride()

    input_numel = input.numel()
    other_numel = other.numel() if isinstance(other, torch.Tensor) else 1
    output_numel = out.numel()

    # Determine the block size
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(output_numel)))
    grid_size = triton.cdiv(output_numel, block_size)

    # Launch the Triton kernel
    add_kernel[(grid_size, 1, 1)](
        input.data_ptr(), other.data_ptr() if isinstance(other, torch.Tensor) else other.item(),
        out.data_ptr(), alpha.item(),
        input_shape, other_shape, output_shape,
        input_strides, other_strides, output_strides,
        input_numel, other_numel, output_numel,
        BLOCK_SIZE=block_size
    )

    return out
