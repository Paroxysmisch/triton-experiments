import torch
import triton
import triton.language as tl

@triton.jit
def _log_tanh_kernel(input_ptr, output_ptr, input_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_size
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.tanh(tl.log(x))
    tl.store(output_ptr + offsets, y, mask=mask)

def log_tanh(input, out=None) -> torch.Tensor:
    assert input.is_cuda, "Input tensor must be on CUDA"
    if not torch.all(input > 0):
        raise ValueError("All input elements must be positive")
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input"
        assert out.is_cuda, "Output tensor must be on CUDA"
    size = input.numel()
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    _log_tanh_kernel[grid](input.data_ptr(), out.data_ptr(), size, BLOCK_SIZE=1024)
    return out
