import triton
import triton.language as tl
import torch

@triton.jit
def _dequantize_rowwise(
    x_ptr,          # pointer to input quantized data [M, N]
    state_x_ptr,    # pointer to scale values [M, 1]
    output_ptr,     # pointer to output buffer [M, N]
    M,              # number of rows
    N,              # number of columns
    stride_xm,      # stride for x matrix rows
    stride_xn,      # stride for x matrix columns
    stride_sm,      # stride for scale matrix rows
    stride_om,      # stride for output matrix rows
    stride_on,      # stride for output matrix columns
    BLOCK_SIZE: tl.constexpr,  # size of block for processing
):
    # Program ID
    pid = tl.program_id(0)  # row index
    
    # Calculate the offsets
    x_offset = pid * stride_xm
    out_offset = pid * stride_om
    
    # Load the scale value for this row
    scale = tl.load(state_x_ptr + pid * stride_sm)
    
    # Calculate number of blocks needed
    n_blocks = tl.cdiv(N, BLOCK_SIZE)
    
    # Constants
    INV_127 = 1.0 / 127.0
    
    # Process the row in blocks
    for block in range(n_blocks):
        # Calculate offsets for this block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        block_offset = block * BLOCK_SIZE
        
        # Mask for handling boundary conditions
        mask = col_offsets + block_offset < N
        
        # Load quantized values
        x = tl.load(x_ptr + x_offset + (block_offset + col_offsets) * stride_xn, mask=mask)
        
        # Convert to float and dequantize
        x = x.to(tl.float32)
        output = x * scale * INV_127
        
        # Store the result
        tl.store(output_ptr + out_offset + (block_offset + col_offsets) * stride_on, 
                output, mask=mask)

def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for row-wise dequantization using Triton kernel.
    
    Args:
        x: Input quantized tensor of shape [M, N]
        state_x: Scale values tensor of shape [M, 1]
    
    Returns:
        Dequantized tensor of shape [M, N]
    """
    M, N = x.shape
    
    # Create output tensor
    output = torch.empty_like(x, dtype=torch.float32, device=x.device)
    
    # Calculate the nearest power of 2 for block size
    BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)  # Cap at 1024 for practical purposes
    
    # Get strides for all tensors
    stride_xm, stride_xn = x.stride()
    stride_sm = state_x.stride(0)
    stride_om, stride_on = output.stride()
    
    # Launch kernel
    grid = (M,)  # One thread block per row
    
    _dequantize_rowwise[grid](
        x_ptr=x, 
        state_x_ptr=state_x,
        output_ptr=output,
        M=M, 
        N=N,
        stride_xm=stride_xm,
        stride_xn=stride_xn,
        stride_sm=stride_sm,
        stride_om=stride_om,
        stride_on=stride_on,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
