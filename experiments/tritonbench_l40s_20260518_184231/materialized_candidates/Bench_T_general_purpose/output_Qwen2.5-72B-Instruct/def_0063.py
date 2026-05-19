import triton
import triton.language as tl

@triton.jit
def tensordot_kernel(
    a_ptr, b_ptr, r_ptr,
    a_shape, b_shape, r_shape,
    a_strides, b_strides, r_strides,
    a_dims, b_dims, r_dims,
    num_contracting_dims,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the current block index
    pid = tl.program_id(0)
    
    # Compute the total number of elements in the result tensor
    total_elements = 1
    for dim in r_shape:
        total_elements *= dim
    
    # Compute the block index and the start index for the block
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, total_elements)
    
    # Initialize the result block
    r_block = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Iterate over the block
    for i in range(block_start, block_end):
        # Compute the indices for the result tensor
        r_indices = [0] * len(r_shape)
        temp = i
        for dim in reversed(range(len(r_shape))):
            r_indices[dim] = temp % r_shape[dim]
            temp //= r_shape[dim]
        
        # Initialize the sum for the current result element
        sum_val = 0.0
        
        # Iterate over the contracting dimensions
        for k in range(num_contracting_dims):
            # Compute the indices for the a and b tensors
            a_indices = [0] * len(a_shape)
            b_indices = [0] * len(b_shape)
            
            for dim in range(len(a_shape)):
                if dim in a_dims:
                    a_indices[dim] = r_indices[a_dims.index(dim)]
                else:
                    a_indices[dim] = r_indices[dim]
            
            for dim in range(len(b_shape)):
                if dim in b_dims:
                    b_indices[dim] = r_indices[b_dims.index(dim)]
                else:
                    b_indices[dim] = r_indices[dim]
            
            # Compute the linear indices for the a and b tensors
            a_index = 0
            b_index = 0
            for dim in range(len(a_shape)):
                a_index += a_indices[dim] * a_strides[dim]
            for dim in range(len(b_shape)):
                b_index += b_indices[dim] * b_strides[dim]
            
            # Load the elements from the a and b tensors
            a_val = tl.load(a_ptr + a_index)
            b_val = tl.load(b_ptr + b_index)
            
            # Compute the product and add to the sum
            sum_val += a_val * b_val
        
        # Store the result in the result tensor
        r_block[i - block_start] = sum_val
    
    # Store the result block back to the result tensor
    tl.store(r_ptr + block_start, r_block)

import torch
import triton
import triton.language as tl
from typing import Union, Tuple, List

def tensordot(a: torch.Tensor, b: torch.Tensor, dims: Union[int, Tuple[List[int], List[int]], List[List[int]]]) -> torch.Tensor:
    # Convert dims to the appropriate format
    if isinstance(dims, int):
        a_dims = list(range(-1, -dims-1, -1))
        b_dims = list(range(dims))
    elif isinstance(dims, (tuple, list)) and len(dims) == 2:
        a_dims, b_dims = dims
    else:
        raise ValueError("Invalid dims format. Expected int, Tuple[List[int], List[int]], or List[List[int]].")
    
    # Validate the dimensions
    if len(a_dims) != len(b_dims):
        raise ValueError("The number of dimensions to contract must be the same for both tensors.")
    
    for dim_a, dim_b in zip(a_dims, b_dims):
        if a.shape[dim_a] != b.shape[dim_b]:
            raise ValueError("The sizes in the contracted dimensions must match.")
    
    # Compute the shape of the result tensor
    r_shape = []
    for i in range(len(a.shape)):
        if i not in a_dims:
            r_shape.append(a.shape[i])
    for i in range(len(b.shape)):
        if i not in b_dims:
            r_shape.append(b.shape[i])
    
    # Compute the strides for the result tensor
    r_strides = [1]
    for dim in reversed(r_shape[:-1]):
        r_strides.insert(0, r_strides[0] * dim)
    
    # Allocate the result tensor
    r = torch.zeros(r_shape, device=a.device, dtype=a.dtype)
    
    # Launch the Triton kernel
    grid = (r.numel() // 1024 + 1,)
    tensordot_kernel[grid](
        a.data_ptr(), b.data_ptr(), r.data_ptr(),
        a.shape, b.shape, r.shape,
        a.stride(), b.stride(), r_strides,
        a_dims, b_dims, list(range(len(r_shape))),
        len(a_dims),
        BLOCK_SIZE=1024,
    )
    
    return r
