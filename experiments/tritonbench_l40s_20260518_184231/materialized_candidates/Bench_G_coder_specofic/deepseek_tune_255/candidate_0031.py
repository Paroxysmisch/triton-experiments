import triton
import triton.language as tl
import torch

@triton.jit
def mul_kernel(src, dst, BLOCK_SIZE: tl.constexpr):
    # Define the exponent compensator
    EXPONENT_COMPENSATOR = 2.0 ** (127 - 15)

    # Compute indices for the block
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load elements from src, apply the compensator, and store in dst
    src_val = tl.load(src + idx)
    result = src_val * EXPONENT_COMPENSATOR
    tl.store(dst + idx, result)

def launch_mul_kernel(src: torch.Tensor, BLOCK_SIZE=128):
    # Create an empty tensor on the CUDA device for the result
    dst = torch.empty_like(src, device="cuda")

    # Launch the Triton kernel
    grid = (src.shape[0] // BLOCK_SIZE,)
    mul_kernel[grid](src, dst, BLOCK_SIZE)

    # Return the filled dst tensor
    return dst
