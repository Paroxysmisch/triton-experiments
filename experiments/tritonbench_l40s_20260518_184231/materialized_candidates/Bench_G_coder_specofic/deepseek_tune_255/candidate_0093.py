import torch
import triton
import triton.language as tl

# Triton kernel for converting float8 to float16
@triton.jit
def kernel_f8_to_f16(
    X,
    Y,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, x, mask=mask)

def f8_to_f16(x: torch.Tensor) -> torch.Tensor:
    assert x.type == torch.int8 and x.is_cuda
    y = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    dtype = triton.get_datatype(x.dtype)
    kernel_f8_to_f16[grid](x, y, n_elements, BLOCK_SIZE=1, dtype=dtype)
    return y

# Triton kernel for converting float16 to float8
@triton.jit
def kernel_f16_to_f8(
    X,
    Y,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, x, mask=mask)

def f16_to_f8(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype in [torch.float16, torch.float32] and x.is_cuda
    y = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    dtype = triton.get_datatype(x.dtype)
    kernel_f16_to_f8[grid](x, y, n_elements, BLOCK_SIZE=1, dtype=dtype)
    return y

def test_triton():
    shape = (128 * 1024 * 1024, )
    x_f8 = torch.randint(low=-128, high=127, size=shape, device="cuda", dtype=torch.int8)
    x_f16 = torch.randn(shape, device="cuda", dtype=torch.float16)
    y_f8 = f16_to_f8(x_f16)
    y_f16 = f8_to_f16(x_f8)
    assert torch.allclose(x_f16, y_f8, atol=1e-2, rtol=0)
    assert torch.allclose(x_f8, y_f16, atol=0, rtol=0)

if __name__ == "__main__":
    test_triton()
