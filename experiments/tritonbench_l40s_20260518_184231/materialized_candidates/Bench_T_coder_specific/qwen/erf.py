import triton
import triton.language as tl
import torch

@torch.jit.script
def erf(input, out=None):
    if out is None:
        out = torch.empty_like(input)

    assert input.device.type == 'cuda', "Input must be on CUDA device"
    assert input.dtype == torch.float32, "Input must be float32"

    grid_size = (len(input) + 127) // 128
    erf_kernel[grid_size, 128](input.data_ptr(), out.data_ptr(), len(input))

    return out
