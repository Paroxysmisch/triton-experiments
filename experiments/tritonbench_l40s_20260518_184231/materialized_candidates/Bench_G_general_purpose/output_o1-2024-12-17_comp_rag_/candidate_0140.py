import triton
import triton.language as tl
import torch

# Kernel from Document 1
@triton.jit
def kernel(X, stride_xm,
           Z, stride_zn,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    off_m = tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    Xs = X + off_m[:, None] * stride_xm + off_n[None, :] * 1
    Zs = Z + off_m[:, None] * 1 + off_n[None, :] * stride_zn
    tl.store(Zs, tl.load(Xs))

ret = triton.compile(
    kernel,
    signature="*fp32,i32,*fp32,i32",
    constants={"BLOCK_M": 64, "BLOCK_N": 64}
)
print(ret.asm["ttgir"])


# Kernel for multiplying by 2
@triton.jit
def mul2_kernel(X_ptr, OUT_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(X_ptr + offsets, mask=mask)
    x = x * 2.0
    tl.store(OUT_ptr + offsets, x, mask=mask)

# Kernel for multiplying by 2 inplace
@triton.jit
def mul2_inplace_kernel(X_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(X_ptr + offsets, mask=mask)
    x = x * 2.0
    tl.store(X_ptr + offsets, x, mask=mask)

def triton_mul2(in_data: torch.Tensor) -> torch.Tensor:
    assert in_data.is_cuda, "Input tensor must be on CUDA device"
    n_elements = in_data.numel()
    out_data = torch.empty_like(in_data)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul2_kernel[grid](
        in_data, 
        out_data, 
        n_elements, 
        BLOCK_SIZE=1024
    )
    return out_data

def triton_mul2_inplace(in_data: torch.Tensor):
    assert in_data.is_cuda, "Input tensor must be on CUDA device"
    n_elements = in_data.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul2_inplace_kernel[grid](
        in_data,
        n_elements,
        BLOCK_SIZE=1024
    )
