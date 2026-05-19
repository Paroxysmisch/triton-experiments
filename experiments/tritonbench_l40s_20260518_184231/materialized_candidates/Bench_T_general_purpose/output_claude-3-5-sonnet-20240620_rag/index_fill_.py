import torch
import triton
import triton.language as tl

@triton.jit
def index_fill_kernel(
    x_ptr,          # Pointer to input/output tensor
    index_ptr,      # Pointer to index tensor
    value,          # Value to fill
    n_indices,      # Number of indices
    stride_dim,     # Stride of the dimension being filled
    stride_other,   # Stride of the other dimension
    n_rows,         # Number of rows
    n_cols,         # Number of columns
    dim: tl.constexpr,  # Dimension along which to fill (0 for rows, 1 for columns)
    BLOCK_SIZE: tl.constexpr,  # Number of elements to process per block
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Calculate starting offset for this program instance
    offset = pid * BLOCK_SIZE
    
    # Create a mask for valid indices
    mask = offset + tl.arange(0, BLOCK_SIZE) < n_indices
    
    # Load indices
    indices = tl.load(index_ptr + offset, mask=mask)
    
    # Process based on dimension
    if dim == 0:  # Fill along rows
        # For each valid index
        for i in range(0, BLOCK_SIZE):
            if i + offset < n_indices:
                idx = tl.load(index_ptr + offset + i)
                # Fill entire row at idx
                row_ptr = x_ptr + idx * stride_dim
                for col in range(0, n_cols):
                    tl.store(row_ptr + col * stride_other, value)
    else:  # Fill along columns
        # For each valid index
        for i in range(0, BLOCK_SIZE):
            if i + offset < n_indices:
                idx = tl.load(index_ptr + offset + i)
                # Fill entire column at idx
                col_ptr = x_ptr + idx * stride_other
                for row in range(0, n_rows):
                    tl.store(col_ptr + row * stride_dim, value)

def index_fill_(x: torch.Tensor, dim: int, index: torch.Tensor, value: float) -> torch.Tensor:
    """
    Fills elements of the tensor with value at positions given by index in dimension dim.
    
    Args:
        x: Input tensor to be modified in-place
        dim: Dimension along which to index
        index: LongTensor containing indices to fill
        value: Value to fill with
    
    Returns:
        Modified tensor
    """
    assert dim < x.dim(), f"Dimension out of range (expected to be < {x.dim()}, but got {dim})"
    assert index.dim() == 1, "Index tensor must be 1-D"
    
    # Move tensors to CUDA if not already
    device = x.device
    if not device.type == 'cuda':
        x = x.cuda()
    if not index.device.type == 'cuda':
        index = index.cuda()
    
    # Get tensor properties
    n_indices = index.numel()
    n_rows = x.size(0)
    n_cols = x.size(1)
    
    # Configure block size and grid
    BLOCK_SIZE = 32
    grid = lambda meta: (triton.cdiv(n_indices, BLOCK_SIZE),)
    
    # Launch kernel
    index_fill_kernel[grid](
        x_ptr=x,
        index_ptr=index,
        value=value,
        n_indices=n_indices,
        stride_dim=x.stride(dim),
        stride_other=x.stride(1 if dim == 0 else 0),
        n_rows=n_rows,
        n_cols=n_cols,
        dim=dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return x

# Example usage and test
def test_index_fill():
    # Create test tensor
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, device='cuda')
    index = torch.tensor([0, 2], device='cuda')
    
    # Test column-wise fill
    x_copy = x.clone()
    index_fill_(x_copy, 1, index, -1)
    
    # Verify result
    expected = torch.tensor([
        [-1, 2, -1],
        [-1, 5, -1],
        [-1, 8, -1]
    ], dtype=torch.float32, device='cuda')
    
    assert torch.allclose(x_copy, expected), "Test failed!"
    print("Test passed!")
