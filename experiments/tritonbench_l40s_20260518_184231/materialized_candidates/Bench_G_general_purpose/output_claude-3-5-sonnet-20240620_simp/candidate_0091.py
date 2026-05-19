import torch
import triton
import triton.language as tl
import math

# Quantization helper functions
def quantize_int4(w: torch.Tensor, scale: float = None) -> torch.Tensor:
    """Quantize weights to INT4 format"""
    if scale is None:
        scale = torch.max(torch.abs(w)).item() / 7.0
    
    # Quantize to integers between -8 and 7
    w_int = torch.clamp(torch.round(w / scale), -8, 7)
    
    # Pack two INT4 values into one INT8
    w_int = w_int.view(-1)
    if w_int.shape[0] % 2 == 1:
        w_int = torch.cat([w_int, torch.zeros(1, dtype=w_int.dtype, device=w_int.device)])
    
    even = w_int[::2]
    odd = w_int[1::2]
    packed = (even + 8) | ((odd + 8) << 4)
    
    return packed.to(torch.int8), scale

def unpack_int4(packed: torch.Tensor, scale: float, original_shape: tuple) -> torch.Tensor:
    """Unpack INT4 weights back to floating point for verification"""
    unpacked = torch.zeros(packed.shape[0] * 2, dtype=torch.float32, device=packed.device)
    unpacked[::2] = ((packed & 0xF) - 8).float()
    unpacked[1::2] = ((packed >> 4) - 8).float()
    return (unpacked[:math.prod(original_shape)].reshape(original_shape) * scale)

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    # Scale for dequantization
    scale,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matrix multiplication C = A x B where B is quantized"""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create block pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate to compute a block of the C matrix
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load and dequantize the B matrix
        b_packed = tl.load(b_ptrs)
        b_even = (b_packed & 0xF).to(tl.float32)
        b_odd = (b_packed >> 4).to(tl.float32)
        b_even = (b_even - 8) * scale
        b_odd = (b_odd - 8) * scale
        
        # Load the A matrix
        a = tl.load(a_ptrs)
        
        # Compute matrix multiplication
        accumulator += tl.dot(a, b_even)
        
        # Move pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        
    # Write back the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def matmul_dequantize_int4_s2(a: torch.Tensor, b_packed: torch.Tensor, scale: float):
    """Wrapper function for the Triton kernel"""
    # Extract dimensions
    M, K = a.shape
    K_div_2, N = b_packed.shape  # Note: K dimension is halved due to INT4 packing
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    GROUP_SIZE_M = 8
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    matmul_kernel[grid](
        a, b_packed, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b_packed.stride(0), b_packed.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        scale=scale,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return c
