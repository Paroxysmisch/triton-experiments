import triton
import triton.language as tl

@triton.jit
def fused_gather_masked_fill_kernel(
    input_ptr, index_ptr, mask_ptr, out_ptr, value, 
    dim, num_elements, num_dims, input_shape_ptr, index_shape_ptr
):
    pid = tl.program_id(0)
    
    # Compute the offset for the current element
    offsets = tl.arange(0, num_elements)
    
    # Load input shape and index shape
    input_shape = tl.load(input_shape_ptr + tl.arange(0, num_dims))
    index_shape = tl.load(index_shape_ptr + tl.arange(0, num_dims))
    
    # Compute the multi-dimensional index
    multi_index = tl.zeros([num_dims], dtype=tl.int32)
    remaining = pid
    for d in range(num_dims - 1, -1, -1):
        if d == dim:
            multi_index[d] = tl.load(index_ptr + remaining % index_shape[d])
        else:
            multi_index[d] = remaining % input_shape[d]
        remaining = remaining // input_shape[d]
    
    # Compute the linear index in the input tensor
    linear_index = tl.zeros([1], dtype=tl.int32)
    stride = 1
    for d in range(num_dims - 1, -1, -1):
        linear_index += multi_index[d] * stride
        stride *= input_shape[d]
    
    # Gather the value from the input tensor
    gathered_value = tl.load(input_ptr + linear_index)
    
    # Apply the mask
    mask_value = tl.load(mask_ptr + pid)
    result = tl.where(mask_value, value, gathered_value)
    
    # Store the result
    tl.store(out_ptr + pid, result)

import torch

def fused_gather_masked_fill(input, dim, index, mask, value, *, sparse_grad=False, out=None):
    # Ensure input and index have the same number of dimensions
    assert input.dim() == index.dim(), "Input and index must have the same number of dimensions."
    
    # Calculate the shape of the output tensor
    output_shape = list(input.shape)
    output_shape[dim] = index.shape[dim]
    
    # Broadcast the mask to the output shape
    mask = mask.expand(output_shape)
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Get the number of elements and dimensions
    num_elements = out.numel()
    num_dims = input.dim()
    
    # Launch the Triton kernel
    fused_gather_masked_fill_kernel[(num_elements,)](
        input_ptr=input.data_ptr(),
        index_ptr=index.data_ptr(),
        mask_ptr=mask.data_ptr(),
        out_ptr=out.data_ptr(),
        value=value,
        dim=dim,
        num_elements=num_elements,
        num_dims=num_dims,
        input_shape_ptr=torch.tensor(input.shape, dtype=torch.int32).data_ptr(),
        index_shape_ptr=torch.tensor(index.shape, dtype=torch.int32).data_ptr()
    )
    
    return out
