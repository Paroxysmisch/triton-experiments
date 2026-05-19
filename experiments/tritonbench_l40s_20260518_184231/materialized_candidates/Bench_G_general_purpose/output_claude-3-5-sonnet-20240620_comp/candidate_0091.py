import torch
import triton
import triton.language as tl
import math

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
    # INT4 quantization parameters
    scales_ptr, zero_points_ptr,
    GROUP_SIZE: tl.constexpr,
):
    """
    Kernel for computing the matrix multiplication C = A x B
    where A is fp32 and B is quantized to INT4
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    
    # Calculate current block indices
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Calculate offsets for the block
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over k dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load scales and zero points for current block
        scale_idx = k * BLOCK_SIZE_K // GROUP_SIZE
        scales = tl.load(scales_ptr + scale_idx)
        zero_points = tl.load(zero_points_ptr + scale_idx)
        
        # Load and unpack INT4 values
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + k * BLOCK_SIZE_K * stride_ak)
        b_packed = tl.load(b_ptr + k * BLOCK_SIZE_K * stride_bk + offs_bn[None, :] * stride_bn)
        
        # Unpack INT4 values (2 values per byte)
        b_unpacked = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
        for i in range(0, 8):  # 8 INT4 values per INT32
            mask = 0xF << (i * 4)
            values = (b_packed & mask) >> (i * 4)
            # Dequantize
            b_unpacked = b_unpacked + (values - zero_points) * scales
        
        # Compute matrix multiplication
        acc += tl.dot(a, b_unpacked)
    
    # Store output
    c = acc.to(tl.float32)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, c)

def quantize_int4(weight_matrix, group_size=128):
    """
    Quantize a weight matrix to INT4 format
    Returns: quantized weights, scales, and zero points
    """
    # Reshape matrix for group quantization
    orig_shape = weight_matrix.shape
    num_groups = math.ceil(weight_matrix.numel() / group_size)
    padded = torch.nn.functional.pad(
        weight_matrix.reshape(-1),
        (0, num_groups * group_size - weight_matrix.numel())
    )
    groups = padded.reshape(-1, group_size)
    
    # Calculate scales and zero points
    min_vals, _ = groups.min(dim=1, keepdim=True)
    max_vals, _ = groups.max(dim=1, keepdim=True)
    scales = (max_vals - min_vals) / 15  # 15 is the range for INT4 (-8 to 7)
    zero_points = torch.round((min_vals / scales) + 8)
    
    # Quantize
    quantized = torch.round(groups / scales + zero_points)
    quantized = torch.clamp(quantized, 0, 15)
    
    # Pack INT4 values (8 values per INT32)
    packed = torch.zeros((num_groups, group_size // 8), dtype=torch.int32)
    for i in range(8):
        packed = packed | (quantized[:, i::8].to(torch.int32) << (i * 4))
    
    return packed.reshape(orig_shape[0], -1), scales.squeeze(), zero_points.squeeze()

def matmul_dequantize_int4_s2(a, b_quantized, scales, zero_points, group_size=128):
    """
    Compute matrix multiplication with quantized INT4 weights
    """
    M, K = a.shape
    _, N = b_quantized.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Grid and block sizes
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        a, b_quantized, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b_quantized.stride(0), b_quantized.stride(1),
        c.stride(0), c.stride(1),
        scales, zero_points,
        GROUP_SIZE=group_size,
    )
    
    return c

def unpack_int4(packed_matrix, scales, zero_points, group_size=128):
    """
    Unpack INT4 matrix back to fp32 (for testing)
    """
    num_groups = packed_matrix.shape[0]
    unpacked = torch.zeros((num_groups, group_size), device=packed_matrix.device)
    
    for i in range(8):
        mask = 0xF << (i * 4)
        values = (packed_matrix & mask) >> (i * 4)
        unpacked[:, i::8] = (values - zero_points.unsqueeze(1)) * scales.unsqueeze(1)
    
    return unpacked
