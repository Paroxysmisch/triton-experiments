import math
import torch
import triton
import triton.language as tl

# Triton kernel for row-wise quantization
@triton.jit
def _quantize_int8_perrow(
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

def quantize_int8_perrow(x: torch.Tensor):
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
    _quantize_int8_perrow[grid](x, output, output_maxs, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
    return output, output_maxs

# Triton kernel for matrix multiplication with quantized matrices
@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_row_maxs,
    b_col_maxs,
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
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_warp = num_pid_m * num_pid_n
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n
    rm, rn = pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        # Load the input elements
        a = tl.load(a_ptr + rm * stride_am + k * stride_ak + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_am + tl.arange(0, BLOCK_SIZE_K)[None, :], mask=rm + tl.arange(0, BLOCK_SIZE_M)[:, None] < M and k + tl.arange(0, BLOCK_SIZE_K)[None, :] < K, other=0)
        b = tl.load(b_ptr + k * stride_bk + rn * stride_bn + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bk + tl.arange(0, BLOCK_SIZE_N)[None, :], mask=k + tl.arange(0, BLOCK_SIZE_K)[:, None] < K and rn + tl.arange(0, BLOCK_SIZE_N)[None, :] < N, other=0)

        # Scale the input elements
        a_row_max = tl.load(a_row_maxs + pid_m)
        b_col_max = tl.load(b_col_maxs + pid_n)
        a = a * (a_row_max / 127.0)
        b = b * (b_col_max / 127.0)

        # Perform the matrix multiplication
        acc += tl.dot(a, b)

    # Store the result
    c = tl.load(c_ptr + rm * stride_cm + rn * stride_cn + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_cm + tl.arange(0, BLOCK_SIZE_N)[None, :], mask=rm + tl.arange(0, BLOCK_SIZE_M)[:, None] < M and rn + tl.arange(0, BLOCK_SIZE_N)[None, :] < N, other=0)
    c += acc
    tl.store(c_ptr + rm * stride_cm + rn * stride_cn + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_cm + tl.arange(0, BLOCK_SIZE_N)[None, :], c, mask=rm + tl.arange(0, BLOCK_SIZE_M)[:, None] < M and rn + tl.arange(0, BLOCK_SIZE_N)[None, :] < N)

def matmul_quantize_int8(a: torch.Tensor, b: torch.Tensor):
    a_quant, a_maxs = quantize_int8_perrow(a)
    b_quant, b_maxs = quantize_int8_perrow(b.T).T  # Transpose b for row-wise quantization
    c = torch.zeros((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)

    # Define grid configuration
    grid = lambda meta: (triton.cdiv(a.shape[0], meta['BLOCK_SIZE_M']) * triton.cdiv(b.shape[1], meta['BLOCK_SIZE_N']),)
    # Launch the Triton kernel
    _matmul_kernel[grid](a_quant, b_quant, c, a_maxs, b_maxs, a.shape[0], b.shape[1], a.shape[1], a_quant.stride(0), a_quant.stride(1), b_quant.stride(0), b_quant.stride(1), c.stride(0), c.stride(1), BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16)
    return c

def matmul_int8(a: torch.Tensor, b: torch.Tensor, a_maxs: torch.Tensor, b_maxs: torch.Tensor):
    c = torch.zeros((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)

    # Define grid configuration
    grid = lambda meta: (triton.cdiv(a.shape[0], meta['BLOCK_SIZE_M']) * triton.cdiv(b.shape[1], meta['BLOCK_SIZE_N']),)
    # Launch the Triton kernel
    _matmul_kernel[grid](a, b, c, a_maxs, b_maxs, a.shape[0], b.shape[1], a.shape[1], a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16)
    return c

def quantize_int8(x: torch.Tensor, axis: int = 0):
    if axis == 0:
        return quantize_int8_perrow(x)
    elif axis == 1:
        return quantize_int8_perrow(x.T)
    else:
        raise ValueError("Axis must be 0 or 1")
