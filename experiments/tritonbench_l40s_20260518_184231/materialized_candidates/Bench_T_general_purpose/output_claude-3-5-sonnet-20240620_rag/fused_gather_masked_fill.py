import torch
import triton
import triton.language as tl

@triton.jit
def fused_gather_masked_fill_kernel(
    output_ptr, input_ptr, index_ptr, mask_ptr, 
    input_strides, index_strides, mask_strides, output_strides,
    dim, value, shape_size, BLOCK_SIZE: tl.constexpr
):
    # Get the linear index for this thread
    pid = tl.program_id(0)
    
    # Calculate multi-dimensional indices
    indices = []
    remaining = pid
    for i in range(shape_size-1, -1, -1):
        if i == dim:
            # Use the provided index for the gather dimension
            idx = tl.load(index_ptr + remaining * index_strides[i])
        else:
            # Calculate index for other dimensions
            idx = remaining % shape_size
            remaining = remaining // shape_size
        indices.append(idx)
    
    # Calculate input offset
    input_offset = 0
    for i, idx in enumerate(indices):
        input_offset += idx * input_strides[i]
    
    # Load input value using gathered index
    input_val = tl.load(input_ptr + input_offset)
    
    # Calculate mask offset
    mask_offset = 0
    for i, idx in enumerate(indices):
        mask_offset += idx * mask_strides[i]
    
    # Load mask value
    mask = tl.load(mask_ptr + mask_offset)
    
    # Apply masked fill
    output = tl.where(mask, value, input_val)
    
    # Calculate output offset
    output_offset = 0
    for i, idx in enumerate(indices):
        output_offset += idx * output_strides[i]
    
    # Store result
    tl.store(output_ptr + output_offset, output)

def fused_gather_masked_fill(input, dim, index, mask, value, *, sparse_grad=False, out=None):
    """
    Fused operation combining torch.gather and torch.Tensor.masked_fill.
    
    Args:
        input (Tensor): The input tensor X
        dim (int): The dimension along which to index
        index (LongTensor): The indices of elements to gather
        mask (BoolTensor): A boolean mask tensor
        value (float): The value to fill in where mask is True
        sparse_grad (bool, optional): If True, gradient w.r.t. input will be sparse
        out (Tensor, optional): Output tensor
    
    Returns:
        Tensor: The result of gather followed by masked_fill
    """
    # Input validation
    assert input.dim() == index.dim(), "Input and index must have same dimensions"
    assert all(idx_size <= in_size for idx_size, in_size in zip(index.shape, input.shape)), \
        "Index size must not exceed input size except at gather dimension"
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Prepare shape information
    shape_size = input.dim()
    
    # Calculate block size (power of 2 >= total elements)
    total_elements = index.numel()
    BLOCK_SIZE = triton.next_power_of_2(total_elements)
    
    # Launch kernel
    grid = (total_elements,)
    fused_gather_masked_fill_kernel[grid](
        out.data_ptr(),
        input.data_ptr(),
        index.data_ptr(),
        mask.data_ptr(),
        input.stride(),
        index.stride(),
        mask.stride(),
        out.stride(),
        dim,
        value,
        shape_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Example usage:
if __name__ == "__main__":
    # Create sample tensors
    input = torch.randn(2, 3, 4)
    index = torch.randint(0, 4, (2, 3, 4))
    mask = torch.rand(2, 3, 4) > 0.5
    value = 0.0
    
    # Run fused operation
    result = fused_gather_masked_fill(input, 2, index, mask, value)
