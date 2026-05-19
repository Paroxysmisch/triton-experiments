import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel_online_v2(
    output_ptr, input_ptr,
    M, N,
    stride_om, stride_on,
    stride_im, stride_in,
    TILE_N: tl.constexpr
):
    # Position of elements processed by this program instance
    row_idx = tl.program_id(0)
    
    # Compute tile-aligned bounds for N dimension
    n_aligned = tl.prev_multiple_of(N, TILE_N)
    
    # Compute row offset for input and output
    row_start_in = row_idx * stride_im
    row_start_out = row_idx * stride_om
    
    # First pass: find max for numerical stability
    row_max = -float('inf')
    for n in range(0, n_aligned, TILE_N):
        # Load input tile
        cols = n + tl.arange(0, TILE_N)
        mask = cols < N
        x = tl.load(input_ptr + row_start_in + cols * stride_in, mask=mask, other=-float('inf'))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    
    # Handle remaining elements
    if n_aligned < N:
        cols = n_aligned + tl.arange(0, N - n_aligned)
        x = tl.load(input_ptr + row_start_in + cols * stride_in)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    
    # Second pass: compute exponentials and sum
    exp_sum = 0.0
    for n in range(0, n_aligned, TILE_N):
        cols = n + tl.arange(0, TILE_N)
        mask = cols < N
        x = tl.load(input_ptr + row_start_in + cols * stride_in, mask=mask, other=-float('inf'))
        x = tl.exp(x - row_max)
        exp_sum += tl.sum(x, axis=0)
        # Store normalized values
        tl.store(output_ptr + row_start_out + cols * stride_on, x, mask=mask)
    
    # Handle remaining elements
    if n_aligned < N:
        cols = n_aligned + tl.arange(0, N - n_aligned)
        x = tl.load(input_ptr + row_start_in + cols * stride_in)
        x = tl.exp(x - row_max)
        exp_sum += tl.sum(x, axis=0)
        tl.store(output_ptr + row_start_out + cols * stride_on, x)
    
    # Final pass: normalize by sum
    for n in range(0, n_aligned, TILE_N):
        cols = n + tl.arange(0, TILE_N)
        mask = cols < N
        x = tl.load(output_ptr + row_start_out + cols * stride_on, mask=mask)
        x = x / exp_sum
        tl.store(output_ptr + row_start_out + cols * stride_on, x, mask=mask)
    
    # Handle remaining elements
    if n_aligned < N:
        cols = n_aligned + tl.arange(0, N - n_aligned)
        x = tl.load(output_ptr + row_start_out + cols * stride_on)
        x = x / exp_sum
        tl.store(output_ptr + row_start_out + cols * stride_on, x)

def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax over the last dimension of x
    Args:
        x: Input tensor of shape [..., n]
    Returns:
        Output tensor of same shape with softmax applied to last dimension
    """
    M, N = x.shape[-2], x.shape[-1]
    # Allocate output
    output = torch.empty_like(x)
    # TILE_N should be tuned based on the GPU architecture
    TILE_N = 128
    
    # Reshape input and output tensors to 2D
    x_reshaped = x.reshape(-1, N)
    output_reshaped = output.reshape(-1, N)
    M = x_reshaped.shape[0]
    
    # Launch kernel
    grid = (M,)
    softmax_kernel_online_v2[grid](
        output_reshaped, x_reshaped,
        M, N,
        output_reshaped.stride(0), output_reshaped.stride(1),
        x_reshaped.stride(0), x_reshaped.stride(1),
        TILE_N=TILE_N
    )
    
    return output
