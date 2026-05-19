import torch
import triton
import triton.language as tl
import math

@triton.jit
def softmax_kernel_online_v2(
    output_ptr, input_ptr,
    M, N,
    stride_om, stride_on,  # output strides
    stride_im, stride_in,  # input strides
    TILE_N: tl.constexpr,
):
    # Row index
    row_idx = tl.program_id(0)
    
    # Column indices
    col_offsets = tl.arange(0, TILE_N)
    
    # Initialize pointers to input and output
    input_row_ptr = input_ptr + row_idx * stride_im
    output_row_ptr = output_ptr + row_idx * stride_om
    
    # Initialize running maximum and sum
    row_max = -float('inf')
    row_sum = 0.0
    
    # First pass: find maximum
    for n in range(0, N, TILE_N):
        # Load input elements
        mask = col_offsets + n < N
        x = tl.load(input_row_ptr + col_offsets * stride_in + n * stride_in, mask=mask, other=-float('inf'))
        # Update running maximum
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    
    # Second pass: compute exponentials and sum
    for n in range(0, N, TILE_N):
        mask = col_offsets + n < N
        # Load and subtract maximum for numerical stability
        x = tl.load(input_row_ptr + col_offsets * stride_in + n * stride_in, mask=mask, other=-float('inf'))
        x = tl.exp(x - row_max)
        # Update running sum
        row_sum += tl.sum(x, axis=0)
        # Store intermediate results
        tl.store(output_row_ptr + col_offsets * stride_on + n * stride_on, x, mask=mask)
    
    # Third pass: normalize by sum
    for n in range(0, N, TILE_N):
        mask = col_offsets + n < N
        # Load intermediate results
        x = tl.load(output_row_ptr + col_offsets * stride_on + n * stride_on, mask=mask)
        # Normalize
        x = x / row_sum
        # Store final results
        tl.store(output_row_ptr + col_offsets * stride_on + n * stride_on, x, mask=mask)

def prev_multiple_of(a, b):
    return math.floor(a / b) * b

def softmax(x):
    """
    Compute softmax using a custom Triton kernel
    Args:
        x: input tensor of shape (M, N)
    Returns:
        output tensor of shape (M, N)
    """
    M, N = x.shape
    # Allocate output
    out = torch.empty_like(x)
    
    # Configure tile size (power of 2)
    TILE_N = min(triton.next_power_of_2(N), 512)
    
    # Enqueue kernel
    grid = (M,)
    softmax_kernel_online_v2[grid](
        out, x,
        M, N,
        out.stride(0), out.stride(1),
        x.stride(0), x.stride(1),
        TILE_N=TILE_N,
    )
    
    return out
