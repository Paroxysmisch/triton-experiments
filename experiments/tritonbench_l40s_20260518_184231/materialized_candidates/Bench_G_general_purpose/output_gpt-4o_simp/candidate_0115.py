import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(x_ptr, rms_weights_ptr, out_ptr, M, N, K, BLOCK_SIZE: tl.constexpr):
    # Calculate the grid and block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate the offsets for the current block
    offset_m = pid_m * BLOCK_SIZE
    offset_n = pid_n * BLOCK_SIZE
    
    # Create a block of data from the input tensor
    x_block = tl.load(x_ptr + offset_m * N * K + offset_n * K + tl.arange(0, BLOCK_SIZE))
    
    # Compute the square of the elements
    x_squared = x_block * x_block
    
    # Sum the squared values across the K dimension
    sum_x_squared = tl.sum(x_squared, axis=0)
    
    # Compute the RMS
    rms = tl.sqrt(sum_x_squared / K)
    
    # Normalize the block by the RMS
    normalized_block = x_block / rms
    
    # Load the RMS weights
    rms_weights = tl.load(rms_weights_ptr + offset_n * K + tl.arange(0, BLOCK_SIZE))
    
    # Scale the normalized values by the RMS weights
    scaled_block = normalized_block * rms_weights
    
    # Store the result in the output tensor
    tl.store(out_ptr + offset_m * N * K + offset_n * K + tl.arange(0, BLOCK_SIZE), scaled_block)

import torch

def rmsnorm_wrapper(x, rms_weights):
    # Ensure input tensors are contiguous
    x = x.contiguous()
    rms_weights = rms_weights.contiguous()
    
    # Get the dimensions of the input tensor
    M, N, K = x.shape
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Define block size
    BLOCK_SIZE = 128  # Adjust this based on your GPU architecture
    
    # Launch the kernel
    grid = (M // BLOCK_SIZE, N // BLOCK_SIZE)
    rmsnorm_triton[grid](x, rms_weights, out, M, N, K, BLOCK_SIZE)
    
    return out
