import triton
import triton.language as tl
import torch

# Define the Triton kernel
@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, output_ptr, M, N, BLOCK_N_SIZE, **meta):
    # Compute block index
    pid = tl.program_id(0)
    
    # Compute the starting position of the block
    start_n = pid * BLOCK_N_SIZE
    
    # Create offsets for the current block
    offsets_n = start_n + tl.arange(0, BLOCK_N_SIZE)
    
    # Load the input data
    x = tl.load(x_ptr + offsets_n, mask=offsets_n < N, other=0.0)
    
    # Compute mean square
    mean_square = tl.sum(x * x, axis=0) / N
    
    # Compute root mean square
    rms = tl.sqrt(mean_square)
    
    # Normalize the input
    normalized_x = x / rms
    
    # Load the weights
    rms_w = tl.load(rms_w_ptr + offsets_n, mask=offsets_n < N, other=1.0)
    
    # Scale by weights
    scaled_x = normalized_x * rms_w
    
    # Store the result
    tl.store(output_ptr + offsets_n, scaled_x, mask=offsets_n < N)

# Wrapper function
def rmsnorm_triton_wrapper(x, rms_w, BLOCK_N_SIZE=128):
    # Get the shape of the input tensor
    B, M, N = x.shape
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Convert tensors to pointers
    x_ptr = x.data_ptr()
    rms_w_ptr = rms_w.data_ptr()
    output_ptr = output.data_ptr()
    
    # Launch the kernel
    grid = (triton.cdiv(N, BLOCK_N_SIZE),)
    rmsnorm_triton[grid](x_ptr, rms_w_ptr, output_ptr, M, N, BLOCK_N_SIZE, num_warps=4)
    
    return output

# Example usage
B, M, N = 32, 128, 512  # Example dimensions
x = torch.randn(B, M, N, device='cuda')
rms_w = torch.randn(N, device='cuda')

output = rmsnorm_triton_wrapper(x, rms_w)
