import torch
import triton
import triton.language as tl

@triton.jit
def min_kernel(
    # Pointers to input and output tensors
    input_ptr,
    output_values_ptr,
    output_indices_ptr,
    # Shape information
    row_length,
    n_rows,
    # Strides for memory access
    input_row_stride,
    input_col_stride,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Initialize min value and index
    row_start = pid * input_row_stride
    min_val = float('inf')
    min_idx = 0
    
    # Process row elements in blocks
    for i in range(0, row_length, BLOCK_SIZE):
        # Create block mask
        mask = tl.arange(0, BLOCK_SIZE) < (row_length - i)
        
        # Load block of elements
        offset = row_start + i * input_col_stride
        block = tl.load(input_ptr + offset, mask=mask, other=float('inf'))
        
        # Find minimum in current block
        block_min = tl.min(block)
        block_argmin = tl.argmin(block)
        
        # Update global minimum if needed
        if block_min < min_val:
            min_val = block_min
            min_idx = i + block_argmin
    
    # Store results
    tl.store(output_values_ptr + pid, min_val)
    tl.store(output_indices_ptr + pid, min_idx)

def min(input: torch.Tensor, dim: int, keepdim: bool = False, *, out=None) -> tuple[torch.Tensor, torch.Tensor]:
    # Handle negative dimensions
    if dim < 0:
        dim = input.dim() + dim
    
    # Ensure dim is valid
    assert 0 <= dim < input.dim(), f"dim {dim} is out of bounds for tensor of dimension {input.dim()}"
    
    # Move dimension to reduce to the last position for easier processing
    if dim != input.dim() - 1:
        input = input.transpose(dim, -1)
    
    # Get shape information
    shape = input.shape
    n_rows = input.numel() // shape[-1]
    row_length = shape[-1]
    
    # Create output tensors if not provided
    if out is None:
        output_shape = list(shape)
        output_shape[-1] = 1 if keepdim else output_shape.pop()
        values = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        indices = torch.empty(output_shape, dtype=torch.long, device=input.device)
    else:
        values, indices = out
    
    # Launch kernel
    grid = (n_rows,)
    min_kernel[grid](
        input.contiguous(),
        values.reshape(-1),
        indices.reshape(-1),
        row_length,
        n_rows,
        input.stride(-2) if input.dim() > 1 else 0,
        input.stride(-1),
        BLOCK_SIZE=min(128, row_length),
    )
    
    # Move dimension back if needed
    if dim != input.dim() - 1:
        values = values.transpose(dim, -1)
        indices = indices.transpose(dim, -1)
    
    return values, indices
