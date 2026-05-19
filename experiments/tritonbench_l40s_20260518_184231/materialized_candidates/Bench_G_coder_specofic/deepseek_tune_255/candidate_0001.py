import triton
import triton.language as tl
import torch

@triton.jit
def _dequantize_rowwise(
    x_ptr,
    state_x,
    output_ptr,
    inv_127,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    mask = arange < BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=mask)
    max_x = tl.load(state_x + pid)
    output = x * max_x * inv_127
    tl.store(output_ptr + offsets, output, mask=mask)

def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and state_x.is_cuda
    output = torch.empty_like(x)
    P2 = int(2 ** (2 * (i := (x.shape[1] - 1).bit_length() - 1)))
    n_elements = x.numel()
    grid = lambda meta: (x.shape[0],)
    _dequantize_rowwise[grid](x, state_x, output, 1.0 / 127, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
    return output
