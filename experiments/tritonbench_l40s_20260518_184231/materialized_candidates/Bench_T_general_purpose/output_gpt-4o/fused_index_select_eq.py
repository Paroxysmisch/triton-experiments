import triton
import triton.language as tl

@triton.jit
def fused_index_select_eq_kernel(
    input_ptr, index_ptr, other_ptr, out_ptr,
    input_shape_0, input_shape_1, index_shape_0,
    dim, other_is_scalar,
    BLOCK_SIZE: tl.constexpr
):
    # Determine the block indices
    pid = tl.program_id(0)
    
    # Calculate the offset for input and index
    offset_index = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Ensure we don't go out of bounds
    mask = offset_index < index_shape_0
    
    # Load indices
    indices = tl.load(index_ptr + offset_index, mask=mask)
    
    # Prepare input offset based on dimension
    input_offset = indices * (dim == 0) + offset_index * (dim == 1)
    
    # Load input elements based on indices
    selected_elements = tl.load(input_ptr + input_offset, mask=mask)
    
    # Load or broadcast 'other'
    if other_is_scalar:
        other_elements = tl.load(other_ptr)
    else:
        other_elements = tl.load(other_ptr + offset_index, mask=mask)
    
    # Perform the element-wise equality comparison
    result = selected_elements == other_elements
    
    # Store the result
    tl.store(out_ptr + offset_index, result, mask=mask)

import torch

def fused_index_select_eq(input, dim, index, other, *, out=None):
    # Validate inputs
    assert isinstance(input, torch.Tensor), "Input must be a tensor"
    assert isinstance(index, (torch.IntTensor, torch.LongTensor)), "Index must be an IntTensor or LongTensor"
    assert isinstance(other, (torch.Tensor, float, int)), "Other must be a tensor or scalar"
    
    # Determine the shape of the output
    selected_shape = list(input.shape)
    selected_shape[dim] = index.shape[0]
    
    # Handle output tensor
    if out is None:
        out = torch.empty(selected_shape, dtype=torch.bool, device=input.device)
    
    # Prepare pointers for Triton
    input_ptr = input.data_ptr()
    index_ptr = index.data_ptr()
    other_ptr = other.data_ptr() if isinstance(other, torch.Tensor) else torch.tensor(other, device=input.device).data_ptr()
    out_ptr = out.data_ptr()
    
    # Launch the Triton kernel
    grid = (index.numel() + 1023) // 1024  # Grid size
    other_is_scalar = isinstance(other, (float, int))
    
    fused_index_select_eq_kernel[grid](
        input_ptr, index_ptr, other_ptr, out_ptr,
        input.shape[0], input.shape[1], index.shape[0],
        dim, other_is_scalar,
        BLOCK_SIZE=1024
    )
    
    return out
