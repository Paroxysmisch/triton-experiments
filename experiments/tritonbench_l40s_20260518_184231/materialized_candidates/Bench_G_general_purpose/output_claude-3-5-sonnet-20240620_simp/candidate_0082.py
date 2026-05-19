import triton
import triton.language as tl
import torch

@triton.jit
def load_reduce_kernel(
    output_ptr,  # Pointer to output vector [M]
    input_ptr,   # Pointer to input matrix [M, N]
    M,           # Number of rows
    N,           # Number of columns
    stride_m,    # Stride for rows
    stride_n,    # Stride for columns
    BLOCK_M: tl.constexpr,  # Block size for M dimension
    BLOCK_N: tl.constexpr,  # Block size for N dimension
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate the row offset for this program instance
    row_start = pid * BLOCK_M
    
    # Create a mask for valid M indices
    row_mask = row_start + tl.arange(0, BLOCK_M) < M
    
    # Initialize output with minimum value
    output = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    
    # Iterate over N dimension in blocks
    for n in range(0, N, BLOCK_N):
        # Create mask for valid N indices
        col_mask = n + tl.arange(0, BLOCK_N) < N
        
        # Combined mask for both dimensions
        mask = row_mask[:, None] & col_mask[None, :]
        
        # Calculate offsets for this block
        offs_m = row_start + tl.arange(0, BLOCK_M)
        offs_n = n + tl.arange(0, BLOCK_N)
        
        # Create pointer block
        block_ptr = input_ptr + (offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)
        
        # Load data
        x = tl.load(block_ptr, mask=mask, other=float('-inf'))
        
        # Update maximum values
        output = tl.maximum(output, tl.max(x, axis=1))
    
    # Store results
    out_ptr = output_ptr + row_start
    tl.store(out_ptr, output, mask=row_mask)

def load_reduce(x: torch.Tensor) -> torch.Tensor:
    """
    Compute maximum along the second dimension of the input matrix
    
    Args:
        x: Input tensor of shape [M, N]
    Returns:
        Output tensor of shape [M] containing maximum values
    """
    assert x.dim() == 2, "Input tensor must be 2-dimensional"
    M, N = x.shape
    
    # Allocate output tensor
    output = torch.empty(M, device=x.device, dtype=x.dtype)
    
    # Define block sizes
    BLOCK_M = 32
    BLOCK_N = 128
    
    # Calculate grid size
    grid = (triton.cdiv(M, BLOCK_M),)
    
    # Launch kernel
    load_reduce_kernel[grid](
        output_ptr=output._ptr,
        input_ptr=x._ptr,
        M=M,
        N=N,
        stride_m=x.stride(0),
        stride_n=x.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return output

# Test the implementation
def test_load_reduce():
    M, N = 1024, 2048
    x = torch.randn(M, N, device='cuda')
    
    # Compare Triton implementation with PyTorch
    torch_result = torch.max(x, dim=1)[0]
    triton_result = load_reduce(x)
    
    assert torch.allclose(torch_result, triton_result)
    print("Test passed!")

if __name__ == "__main__":
    test_load_reduce()
