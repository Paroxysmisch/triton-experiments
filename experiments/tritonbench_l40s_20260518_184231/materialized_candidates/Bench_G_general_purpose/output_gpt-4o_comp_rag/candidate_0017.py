import math
import torch
import triton
import triton.language as tl

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
    row_mask = arange < BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=row_mask, other=0.0)
    max_val = tl.load(state_x + pid)
    output = max_val * x * inv_127
    tl.store(output_ptr + offsets, output, mask=row_mask)

def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor):
    output = torch.empty(*x.shape, device=x.device, dtype=torch.float16)

    P2 = int(2 ** (math.ceil(math.log2(x.shape[1]))))

    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (x.shape[0],)
    _dequantize_rowwise[grid](x, state_x, output, 1./127, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
    return output
