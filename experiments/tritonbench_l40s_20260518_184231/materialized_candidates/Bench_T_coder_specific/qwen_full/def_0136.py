import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x, y, x_max, exp_x, sum_x, stride, n_elements, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * XBLOCK
    offsets = block_start + tl.arange(0, XBLOCK)
    mask = offsets < n_elements

    # 1. Find the max value of each slice
    x_max = tl.max(tl.load(x + offsets * stride, mask=mask), axis=0)

    # 2. Compute exp(x - x_max)
    tl.store(exp_x + offsets, tl.exp(tl.load(x + offsets * stride, mask=mask) - x_max), mask=mask)

    # 3. Compute the sum of exp(x - x_max)
    sum_x = tl.sum(tl.load(exp_x + offsets, mask=mask), axis=0)

    # 4. Compute the softmax
    softmax_res = tl.load(exp_x + offsets, mask=mask) / sum_x

    # 5. Store the result
    tl.store(y + offsets * stride, softmax_res, mask=mask)

def softmax(x, dim=None, dtype=None):
    if dim is None:
        dim = -1
    if dtype is None:
        dtype = x.dtype

    shape = x.shape
    x = x.contiguous()

    n_elements = x.numel()
    stride = x.stride()
    output = torch.empty_like(x, dtype=dtype)

    grid = lambda meta: (triton.cdiv(n_elements, meta['XBLOCK']), )

    softmax_kernel[grid](x, output, torch.zeros(1, dtype=torch.float32, device=x.device),
                         torch.zeros(1, dtype=torch.float32, device=x.device), 0., n_elements, XBLOCK=1024)

    return output
