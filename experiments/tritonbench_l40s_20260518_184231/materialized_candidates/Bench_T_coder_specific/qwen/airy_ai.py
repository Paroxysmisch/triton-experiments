import torch
import triton

def airy_ai(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    assert input.device.type == 'cuda', "Input must be on CUDA device"
    assert input.dtype == torch.float32, "Input must be float32"

    BLOCK_SIZE = 1024
    grid_size = (input.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE

    airy_ai_kernel[input.numel(), BLOCK_SIZE](input.data_ptr(), out.data_ptr(), input.numel())

    return out
