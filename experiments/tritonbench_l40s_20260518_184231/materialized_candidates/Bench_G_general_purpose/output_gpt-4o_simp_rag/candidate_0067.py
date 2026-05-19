import math
import torch
import triton
import triton.language as tl

# Triton kernel for row-wise quantization to int8
@triton.jit
def quantize_int8_perrow_kernel(
    x_ptr, output_ptr, output_maxs, n_elements, BLOCK_SIZE: tl.constexpr, P2: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    row_mask = arange < BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=row_mask)

    abs_x = tl.abs(x)
    max_val = tl.max(tl.where(row_mask, abs_x, 0), axis=0)
    output = tl.libdevice.llrint(127.0 * (x / max_val))
    tl.store(output_ptr + offsets, output, mask=row_mask)
    tl.store(output_maxs + pid, max_val)

def quantize_int8_perrow(x: torch.Tensor):
    output = torch.empty_like(x, dtype=torch.int8)
    output_maxs = torch.empty(x.shape[0], device=x.device, dtype=torch.float16)

    P2 = int(2 ** (math.ceil(math.log2(x.shape[1]))))

    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (x.shape[0],)
    quantize_int8_perrow_kernel[grid](x, output, output_maxs, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
    return output, output_maxs

# Triton kernel for matrix multiplication
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K, a_scale_ptr, b_scale_ptr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    arange_m = tl.arange(0, BLOCK_SIZE_M)
    arange_n = tl.arange(0, BLOCK_SIZE_N)
    arange_k = tl.arange(0, BLOCK_SIZE_K)

    offs_am = pid_m * BLOCK_SIZE_M + arange_m
    offs_bn = pid_n * BLOCK_SIZE_N + arange_n
    offs_k = arange_k

    a = tl.load(a_ptr + offs_am[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_bn[None, :])

    a_scale = tl.load(a_scale_ptr + offs_am)
    b_scale = tl.load(b_scale_ptr + offs_bn)

    c = tl.dot(a.to(tl.float32) * a_scale[:, None], b.to(tl.float32) * b_scale[None, :])
    tl.store(c_ptr + offs_am[:, None] * N + offs_bn[None, :], c)

def matmul_quantize_int8(a: torch.Tensor, b: torch.Tensor):
    a_quant, a_maxs = quantize_int8_perrow(a)
    b_quant, b_maxs = quantize_int8_perrow(b)

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = lambda meta: (M // meta['BLOCK_SIZE_M'], N // meta['BLOCK_SIZE_N'])
    matmul_kernel[grid](
        a_quant, b_quant, c, M, N, K, a_maxs, b_maxs,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32
    )
    return c

def matmul_int8(a: torch.Tensor, b: torch.Tensor, a_maxs: torch.Tensor, b_maxs: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = lambda meta: (M // meta['BLOCK_SIZE_M'], N // meta['BLOCK_SIZE_N'])
    matmul_kernel[grid](
        a, b, c, M, N, K, a_maxs, b_maxs,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32
    )
    return c

def quantize_int8(x: torch.Tensor, axis: int):
    if axis == 0:
        return quantize_int8_perrow(x)
    else:
        raise NotImplementedError("Only row-wise quantization is implemented.")
