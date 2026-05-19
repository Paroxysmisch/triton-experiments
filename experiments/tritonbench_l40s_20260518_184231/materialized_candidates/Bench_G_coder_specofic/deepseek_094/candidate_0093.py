import triton
import triton.language as tl
import torch

@triton.jit
def kernel_f8_to_f16(X_ptr, Y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id()
    num_warps = tl.cdiv(num_elements, BLOCK_SIZE)
    num_warps = num_warps * tl.cuda.warp_size
    offsets = pid * num_warps
    mask = tl.mask(offsets < num_elements)

    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.to_float16(x)
    tl.store(Y_ptr + offsets, y, mask=mask)

def f8_to_f16(X: torch.Tensor):
    assert X.dtype == torch.int8 and X.device.type == 'cuda'
    Y = torch.empty_like(X, dtype=torch.float16)
    grid = lambda meta: (meta['num_warps'],)
    kernel_f8_to_f16[grid](X, Y, X.numel(), BLOCK_SIZE=256)
    return Y

@triton.jit
def kernel_f16_to_f8(X_ptr, Y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id()
    num_warps = tl.cdiv(num_elements, BLOCK_SIZE)
    num_warps = num_warps * tl.cuda.warp_size
    offsets = pid * num_warps
    mask = tl.mask(offsets < num_elements)

    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.to_int8(x)
    tl.store(Y_ptr + offsets, y, mask=mask)

def f16_to_f8(X: torch.Tensor):
    assert X.dtype == torch.float16 and X.device.type == 'cuda'
    Y = torch.empty_like(X, dtype=torch.int8)
    grid = lambda meta: (meta['num_warps'],)
    kernel_f16_to_f8[grid](X, Y, X.numel(), BLOCK_SIZE=256)
    return Y
