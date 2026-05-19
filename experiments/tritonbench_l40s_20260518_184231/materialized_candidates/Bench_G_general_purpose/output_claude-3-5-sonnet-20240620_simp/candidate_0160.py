import torch
import triton
import triton.language as tl

# Constants for block sizes
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 256
BLOCK_SIZE_K = 32
GROUP_SIZE = 128  # Number of elements per quantization group

@triton.jit
def matmul4_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Quantization parameters
    scales_ptr, zeros_ptr,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    Kernel for computing the matrix multiplication C = A x B where B is quantized.
    A is of shape (M, K) float16
    B is of shape (K, N) int32 (packed int4)
    C is of shape (M, N) float16
    """
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)

    # Block pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate to compute a block of the C matrix
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A and B blocks
        a = tl.load(a_ptrs)
        b_packed = tl.load(b_ptrs)
        
        # Unpack B (int4 values) - 8 values per int32
        b_idx = offs_k[:, None] // GROUP_SIZE
        scale = tl.load(scales_ptr + b_idx)
        zero = tl.load(zeros_ptr + b_idx)
        
        # Extract int4 values and dequantize
        b_shift = (offs_k[:, None] % 8) * 4
        b_mask = 0xF
        b_int4 = (b_packed >> b_shift) & b_mask
        b = (b_int4.to(tl.float32) - zero) * scale

        # Compute matrix multiplication
        accumulator += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def quantize_int4(matrix, group_size=128):
    """
    Quantize a matrix to int4 format using GPTQ method
    Returns: packed int32 tensor, scales, zero_points
    """
    # Reshape for group-wise quantization
    orig_shape = matrix.shape
    matrix = matrix.reshape(-1, group_size)
    
    # Compute scales and zero points
    max_vals = matrix.max(dim=1).values
    min_vals = matrix.min(dim=1).values
    scales = (max_vals - min_vals) / 15
    zeros = min_vals
    
    # Quantize to int4
    matrix_normalized = ((matrix - zeros.unsqueeze(1)) / scales.unsqueeze(1))
    matrix_int4 = torch.clamp(torch.round(matrix_normalized), 0, 15).to(torch.int32)
    
    # Pack 8 int4 values into each int32
    packed = torch.zeros((matrix.shape[0], (matrix.shape[1] + 7) // 8), 
                        dtype=torch.int32, device=matrix.device)
    for i in range(8):
        packed[:, :] |= (matrix_int4[:, i::8] & 0xF) << (i * 4)
    
    return packed.reshape(orig_shape[0], -1), scales, zeros

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}),
    ],
    key=['M', 'N', 'K'],
)
def matmul_dequantize_int4_gptq(a, b_packed, scales, zeros):
    """
    Compute matrix multiplication with quantized weights: C = A × B
    where B is stored in int4 format
    
    Args:
        a: torch.Tensor(M, K) - float16 input matrix
        b_packed: torch.Tensor(K//8, N) - int32 packed matrix (8 int4 values per int32)
        scales: torch.Tensor(K//GROUP_SIZE) - scaling factors
        zeros: torch.Tensor(K//GROUP_SIZE) - zero points
    Returns:
        c: torch.Tensor(M, N) - float16 output matrix
    """
    M, K = a.shape
    K_packed, N = b_packed.shape
    K = K_packed * 8  # Actual K dimension (unpacked)
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # Launch kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * 
                        triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul4_kernel[grid](
        a, b_packed, c,
        M, N, K,
        scales, zeros,
        a.stride(0), a.stride(1),
        b_packed.stride(0), b_packed.stride(1),
        c.stride(0), c.stride(1),
        GROUP_SIZE=GROUP_SIZE
    )
    
    return c
