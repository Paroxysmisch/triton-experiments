import torch
import triton
import triton.language as tl

@triton.jit
def fused_index_select_eq_kernel(
    # Pointers to input, index, other, and output tensors
    input_ptr, index_ptr, other_ptr, output_ptr,
    # Shape and stride information
    n_elements, input_stride_dim, other_stride_dim,
    input_shape_dim, other_shape_dim,
    # Other parameters
    dim, BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load indices
    indices = tl.load(index_ptr + offsets, mask=mask)
    
    # Calculate input offsets based on dimension
    input_offsets = indices * input_stride_dim
    input_vals = tl.load(input_ptr + input_offsets, mask=mask)
    
    # Load other tensor values (handle broadcasting)
    other_vals = tl.load(other_ptr + (offsets % other_shape_dim) * other_stride_dim, mask=mask)
    
    # Perform equality comparison
    result = input_vals == other_vals
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def fused_index_select_eq(input, dim, index, other, *, out=None):
    """
    Performs fused index selection and element-wise equality comparison.
    
    Args:
        input (Tensor): Input tensor
        dim (int): Dimension along which to index
        index (IntTensor or LongTensor): Indices to select along dimension dim
        other (Tensor or float): Tensor or value to compare with
        out (Tensor, optional): Output tensor. Default: None
    
    Returns:
        Tensor: Boolean tensor containing the element-wise equality results
    """
    # Input validation
    if not isinstance(dim, int):
        raise TypeError("dim must be an integer")
    if dim >= input.dim() or dim < -input.dim():
        raise IndexError("dim out of range")
    
    # Convert scalar other to tensor if needed
    if isinstance(other, (int, float)):
        other = torch.tensor(other, device=input.device, dtype=input.dtype)
    
    # Ensure index is on the correct device
    if index.device != input.device:
        index = index.to(input.device)
    
    # Calculate output shape
    output_shape = list(input.shape)
    output_shape[dim] = len(index)
    
    # Initialize output tensor if not provided
    if out is None:
        out = torch.empty(output_shape, dtype=torch.bool, device=input.device)
    
    # Launch kernel
    n_elements = out.numel()
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_index_select_eq_kernel[grid](
        input.data_ptr(),
        index.data_ptr(),
        other.data_ptr(),
        out.data_ptr(),
        n_elements,
        input.stride(dim),
        other.stride(0) if other.dim() > 0 else 0,
        input.shape[dim],
        other.shape[0] if other.dim() > 0 else 1,
        dim,
        BLOCK_SIZE
    )
    
    return out
