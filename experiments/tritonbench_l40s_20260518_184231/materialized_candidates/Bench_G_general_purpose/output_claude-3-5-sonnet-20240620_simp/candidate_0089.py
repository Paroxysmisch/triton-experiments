import torch
import triton
import triton.language as tl

@triton.jit
def _quantize_rowwise_kernel(
    # Pointers to tensors
    input_ptr,      # Pointer to input tensor (float32)
    output_ptr,     # Pointer to output tensor (int8)
    max_vals_ptr,   # Pointer to max values tensor (float32)
    # Shape and stride information
    n_rows,         # Number of rows
    n_cols,         # Number of columns
    input_row_stride,
    output_row_stride,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Each program handles one row
    if pid >= n_rows:
        return
        
    # Compute the start offset for this row
    row_start_in = pid * input_row_stride
    row_start_out = pid * output_row_stride
    
    # Load the row elements in blocks
    max_val = 0.0
    
    # First pass: find maximum absolute value in the row
    for block_start in range(0, n_cols, BLOCK_SIZE):
        # Create a mask for valid elements
        mask = block_start + tl.arange(0, BLOCK_SIZE) < n_cols
        # Load input block
        x = tl.load(input_ptr + row_start_in + block_start, mask=mask, other=0.0)
        # Update running maximum
        max_val = tl.maximum(max_val, tl.maximum(tl.abs(x)))
    
    # Store the max value for this row
    tl.store(max_vals_ptr + pid, max_val)
    
    # Compute scaling factor (127.0 for int8)
    scale = 127.0 / (max_val + 1e-5)  # Add epsilon to avoid division by zero
    
    # Second pass: quantize the elements
    for block_start in range(0, n_cols, BLOCK_SIZE):
        mask = block_start + tl.arange(0, BLOCK_SIZE) < n_cols
        # Load input block
        x = tl.load(input_ptr + row_start_in + block_start, mask=mask, other=0.0)
        # Quantize
        x_quant = tl.math.round(x * scale)
        # Store quantized values
        tl.store(output_ptr + row_start_out + block_start, x_quant, mask=mask)

# Wrapper function
def quantize_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor row-wise to int8.
    
    Args:
        x: Input tensor of shape (n_rows, n_cols)
        
    Returns:
        tuple: (quantized_tensor, max_values)
            - quantized_tensor: int8 tensor of same shape as input
            - max_values: float32 tensor of shape (n_rows,) containing max absolute values
    """
    assert x.dim() == 2, "Input tensor must be 2-dimensional"
    assert x.is_cuda, "Input tensor must be on GPU"
    
    n_rows, n_cols = x.shape
    
    # Create output tensors
    output = torch.empty_like(x, dtype=torch.int8, device=x.device)
    max_vals = torch.empty(n_rows, dtype=torch.float32, device=x.device)
    
    # Configure block sizes
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 1024)
    P2 = max(triton.next_power_of_2(n_cols) // BLOCK_SIZE, 1)
    
    # Launch kernel
    grid = (n_rows,)
    _quantize_rowwise_kernel[grid](
        x, output, max_vals,
        n_rows, n_cols,
        x.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE, P2=P2,
    )
    
    return output, max_vals
