import math
import torch
import triton
import triton.language as tl

# Triton kernel for rowwise quantization
@triton.jit
def _quantize_rowwise(
    x_ptr,
    output_ptr,
    output_maxs,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    # Calculate the block index and the element offsets within the block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    row_mask = arange < BLOCK_SIZE
    # Load the input elements
    x = tl.load(x_ptr + offsets, mask=row_mask)

    # Calculate the absolute maximum value for normalization
    abs_x = tl.abs(x)
    max_val = tl.max(tl.where(row_mask, abs_x, 0), axis=0)
    # Quantize the input elements to int8
    output = tl.libdevice.llrint(127.0 * (x / max_val))
    # Store the quantized output and max values
    tl.store(output_ptr + offsets, output, mask=row_mask)
    tl.store(output_maxs + pid, max_val)

def quantize_rowwise(x: torch.Tensor):
    # Prepare output tensors
    output = torch.empty(*x.shape, device=x.device, dtype=torch.int8)
    output_maxs = torch.empty(x.shape[0], device=x.device, dtype=torch.float16)

    # Calculate the power of two size
    P2 = int(2 ** (math.ceil(math.log2(x.shape[1]))))

    # Ensure CUDA compatibility
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    # Define grid configuration
    grid = lambda meta: (x.shape[0],)
    # Launch the Triton kernel
    _quantize_rowwise[grid](x, output, output_maxs, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
    return output, output_maxs

# Triton kernel for matrix multiplication with quantized int8 matrices
@triton.jit
def _matmul_quantize_int8(
    a_ptr,
    b_ptr,
    c_ptr,
    as_ptr,
    bs_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_warp = num_pid_m * num_pid_n
    num_warp_m = tl.cdiv(BLOCK_SIZE_M, tl.warp_size())
    num_warp_n = tl.cdiv(BLOCK_SIZE_N, tl.warp_size())
    num_warp = num_warp_m * num_warp_n
    warp_id = pid % num_warp
    pid_m = (pid // num_pid_n) % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n
    wm = warp_id % num_warp_m
    wn = warp_id // num_warp_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K * SPLIT_K):
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        a_scale = tl.load(as_ptr + (offs_am // BLOCK_SIZE_M))
        b_scale = tl.load(bs_ptr + (offs_bn // BLOCK_SIZE_N))

        a = a * a_scale[:, None]
        b = b * b_scale[None, :]

        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    c = accumulator
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul_quantize_int8(a: torch.Tensor, b: torch.Tensor, ascales: torch.Tensor, bscales: torch.Tensor):
    assert a.is_cuda and b.is_cuda and ascales.is_cuda and bscales.is_cuda
    assert a.dtype == torch.int8 and b.dtype == torch.int8
    assert ascales.dtype == torch.float16 and bscales.dtype == torch.float16
    assert a.shape[1] == b.shape[0]

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    SPLIT_K = 1

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    _matmul_quantize_int8[grid](
        a, b, c, ascales, bscales,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, SPLIT_K
    )
    return c

# Example usage
M, K, N = 128, 128, 128
fpa = torch.randn((M, K), device='cuda', dtype=torch.float32)
fpb = torch.randn((K, N), device='cuda', dtype=torch.float32)

# Quantize the matrices
a, ascales = quantize_rowwise(fpa)
b, bscales = quantize_rowwise(fpb)

# Perform matrix multiplication
c = matmul_quantize_int8(a, b, ascales, bscales)

print(c)
