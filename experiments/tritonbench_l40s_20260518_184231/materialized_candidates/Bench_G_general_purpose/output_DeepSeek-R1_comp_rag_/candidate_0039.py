import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride,
    n_rows, n_cols, BLOCK_SIZE: tl.constexpr
):
    # Get current row index from program ID
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return  # Early exit for out-of-bounds rows
    
    # Calculate base pointer for current row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # Generate column offsets and mask for valid columns
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    
    # Load row data with masking (invalid elements get -inf for correct max)
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Numerical stability: subtract maximum
    row_max = tl.max(row, axis=0)
    row_minus_max = row - row_max
    
    # Compute softmax numerator and denominator
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # Store results with masking
    output_row_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    
    # Determine optimal block size (next power of two, minimum 32 threads)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = max(BLOCK_SIZE, 32)  # Ensure minimum thread block size
    
    # Calculate number of warps needed (1 warp = 32 threads)
    num_warps = BLOCK_SIZE // 32
    
    # Allocate output tensor
    y = torch.empty_like(x)
    
    # Configure and launch kernel
    grid = (n_rows, 1, 1)  # One kernel instance per row
    softmax_kernel[grid](
        y, x,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return y

# Validation test
torch.manual_seed(42)
x = torch.randn(1823, 781, device='cuda')
y_triton = softmax(x)
y_torch = torch.softmax(x, dim=1)

assert torch.allclose(y_triton, y_torch, atol=1e-4, rtol=1e-4), "Triton and Torch results mismatch"
print("Validation successful!")
