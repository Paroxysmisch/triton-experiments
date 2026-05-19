import triton
import triton.language as tl
import torch
import math

@triton.jit
def bitwise_and_tensor_kernel(
    input_ptr, other_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    output = input & other
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def bitwise_and_scalar_kernel(
    input_ptr, other_scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = input & other_scalar
    tl.store(output_ptr + offsets, output, mask=mask)

def bitwise_and(input, other, *, out=None):
    # Check input tensor validity
    if input.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("input must be an integral or boolean tensor")
    
    # Check other tensor or scalar validity
    if isinstance(other, torch.Tensor):
        if other.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("other must be an integral or boolean tensor")
        if input.shape != other.shape:
            raise ValueError("input and other must have the same shape")
    else:
        if not isinstance(other, (bool, int)):
            raise TypeError("other must be an integer or boolean scalar")
        # Cast Python bool/int to torch scalar tensor for type consistency
        other = torch.tensor(other, device=input.device, dtype=input.dtype)
    
    # Handle output tensor
    if out is not None:
        if out.dtype != input.dtype:
            raise TypeError("out tensor must have the same dtype as input")
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input")
        output = out
    else:
        output = torch.empty_like(input)
    
    n_elements = output.numel()
    if n_elements == 0:
        return output  # No elements to process
    
    # Compute block and grid sizes
    max_block_size = 1024  # Reasonable maximum block size
    block_size = min(triton.next_power_of_2(math.ceil(math.sqrt(n_elements))), max_block_size)
    grid_size = triton.cdiv(n_elements, block_size)
    
    # Dispatch appropriate kernel
    if isinstance(other, torch.Tensor):
        bitwise_and_tensor_kernel[(grid_size, 1, 1)](
            input, other, output, n_elements, BLOCK_SIZE=block_size
        )
    else:
        # Extract scalar value from tensor for Triton
        scalar_value = other.item()
        bitwise_and_scalar_kernel[(grid_size, 1, 1)](
            input, scalar_value, output, n_elements, BLOCK_SIZE=block_size
        )
    
    return output
