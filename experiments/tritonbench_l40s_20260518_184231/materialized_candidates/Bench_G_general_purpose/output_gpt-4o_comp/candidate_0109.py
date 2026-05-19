import triton
import triton.language as tl

@triton.jit
def softmax_kernel_online_v2(input_ptr, output_ptr, M, N, TILE_N, stride_m, stride_n):
    pid = tl.program_id(0)
    
    # Compute the row index for this program instance
    row_idx = pid
    
    # Compute the starting column index for this tile
    col_start = tl.arange(0, TILE_N)
    
    # Load the row from input tensor
    input_row = tl.load(input_ptr + row_idx * stride_m + col_start, mask=col_start < N, other=-float('inf'))
    
    # Compute the maximum value for numerical stability
    row_max = tl.max(input_row, axis=0)
    
    # Compute exponentials and sum them up
    exp_values = tl.exp(input_row - row_max)
    exp_sum = tl.sum(exp_values, axis=0)
    
    # Normalize by the sum to get softmax
    softmax_values = exp_values / exp_sum
    
    # Store the result
    tl.store(output_ptr + row_idx * stride_m + col_start, softmax_values, mask=col_start < N)

def softmax(input_tensor, TILE_N):
    # Get input tensor dimensions
    M, N = input_tensor.shape
    
    # Allocate output tensor
    output_tensor = torch.empty_like(input_tensor)
    
    # Get strides
    stride_m, stride_n = input_tensor.stride()
    
    # Launch the kernel
    grid = (M,)  # Launch one program per row
    softmax_kernel_online_v2[grid](input_tensor, output_tensor, M, N, TILE_N, stride_m, stride_n)
    
    return output_tensor
