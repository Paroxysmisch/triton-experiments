import torch
import triton
import triton.language as tl

# Define the wrapper function
def launch_mul_kernel(src, BLOCK_SIZE=256):
    # Create an empty destination tensor on the CUDA device
    dst = torch.empty_like(src, device='cuda')

    # Launch the Triton kernel
    grid_size = (src.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
    triton.launch_kernel(
        mul_kernel,
        (grid_size,),
        {
            'src': src,
            'dst': dst,
            'BLOCK_SIZE': BLOCK_SIZE
        }
    )

    return dst
