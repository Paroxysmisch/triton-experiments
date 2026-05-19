import triton
import triton.language as tl

@triton.jit
def fused_hstack_div_kernel(
    *input_ptrs,  # Pointers to the input tensors
    divisor_ptr,  # Pointer to the divisor tensor
    output_ptr,   # Pointer to the output tensor
    num_tensors,  # Number of input tensors
    tensor_shape, # Shape of the input tensors (excluding the stacking dimension)
    tensor_stride, # Stride of the input tensors in the stacking dimension
    divisor_shape, # Shape of the divisor tensor
    divisor_stride, # Stride of the divisor tensor
    rounding_mode,  # Rounding mode
    BLOCK_SIZE: tl.constexpr
):
    # Compute the linear index in the output tensor
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load the divisor
    divisor = tl.load(divisor_ptr + (offsets % divisor_shape) * divisor_stride)

    # Initialize the output with zeros
    output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Accumulate the values from the input tensors
    for i in range(num_tensors):
        input_ptr = input_ptrs[i]
        input_offset = i * tensor_stride + offsets
        input_value = tl.load(input_ptr + input_offset)
        output += input_value

    # Perform the division
    if rounding_mode == 0:  # None
        output /= divisor
    elif rounding_mode == 1:  # 'trunc'
        output = tl.where(output >= 0, tl.floor(output / divisor), -tl.floor(-output / divisor))
    elif rounding_mode == 2:  # 'floor'
        output = tl.floor(output / divisor)

    # Store the result
    tl.store(output_ptr + offsets, output)

import torch
import triton
import triton.language as tl

def fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None):
    # Check input types and shapes
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("All elements in 'tensors' must be tensors.")
    if not isinstance(divisor, (torch.Tensor, int, float)):
        raise TypeError("divisor must be a tensor or a number.")
    
    # Convert divisor to tensor if it's a number
    if isinstance(divisor, (int, float)):
        divisor = torch.tensor(divisor, dtype=tensors[0].dtype, device=tensors[0].device)
    
    # Ensure all tensors have compatible shapes for horizontal stacking
    tensor_shape = tensors[0].shape[1:]
    tensor_stride = tensors[0].shape[1] * tensors[0].stride(1)
    for t in tensors[1:]:
        if t.shape[1:] != tensor_shape:
            raise ValueError("All tensors must have the same shape except for the stacking dimension.")
    
    # Ensure the divisor is broadcastable to the shape of the stacked tensor
    stacked_shape = (sum(t.shape[0] for t in tensors),) + tensor_shape
    if not torch.broadcast_shapes(divisor.shape, stacked_shape) == stacked_shape:
        raise ValueError("divisor must be broadcastable to the shape of the stacked tensor.")
    
    # Determine the rounding mode
    rounding_mode_map = {None: 0, 'trunc': 1, 'floor': 2}
    if rounding_mode not in rounding_mode_map:
        raise ValueError("rounding_mode must be one of None, 'trunc', or 'floor'.")
    rounding_mode = rounding_mode_map[rounding_mode]
    
    # Create the output tensor if not provided
    if out is None:
        out = torch.empty(stacked_shape, dtype=tensors[0].dtype, device=tensors[0].device)
    else:
        if out.shape != stacked_shape:
            raise ValueError("out tensor must have the same shape as the stacked tensor.")
    
    # Launch the Triton kernel
    grid = (out.numel() // 1024 + 1,)
    fused_hstack_div_kernel[grid](
        *[t.data_ptr() for t in tensors],
        divisor.data_ptr(),
        out.data_ptr(),
        len(tensors),
        tensors[0].shape[1],
        tensor_stride,
        divisor.shape[0],
        divisor.stride(0),
        rounding_mode,
        BLOCK_SIZE=1024
    )
    
    return out

# Example usage
import torch

# Create some tensors
t1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
t2 = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
divisor = torch.tensor([2, 2], dtype=torch.float32)

# Call the fused_hstack_div function
result = fused_hstack_div([t1, t2], divisor, rounding_mode='floor')

# Print the result
print(result)
