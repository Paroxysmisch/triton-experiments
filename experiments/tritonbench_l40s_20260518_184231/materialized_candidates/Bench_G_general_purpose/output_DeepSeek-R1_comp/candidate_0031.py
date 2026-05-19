import triton
import triton.language as tl
import torch

@triton.jit
def mul_kernel(
    src_ptr,
    dst_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    compensator = 2.0 ** (127 - 15)
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    src_vals = tl.load(src_ptr + offsets)
    dst_vals = src_vals * compensator
    tl.store(dst_ptr + offsets, dst_vals)

def launch_mul_kernel(src: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    assert src.is_cuda, "Input tensor must be on CUDA device"
    dst = torch.empty_like(src)
    grid = (src.shape[0] // BLOCK_SIZE,)
    mul_kernel[grid](src.data_ptr(), dst.data_ptr(), BLOCK_SIZE=BLOCK_SIZE)
    return dst
