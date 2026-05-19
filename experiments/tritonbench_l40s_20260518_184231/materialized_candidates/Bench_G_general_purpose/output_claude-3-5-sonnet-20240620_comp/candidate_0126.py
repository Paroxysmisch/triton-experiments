import torch
import triton
import triton.language as tl

@triton.jit
def matmul4_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    groupsize, NO_GROUPS,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    # Matrix multiplication grid
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    
    # Program ID to block indices
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Block start indices
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over k dimension
    for k in range(0, num_pid_k):
        k_idx = k * BLOCK_SIZE_K
        
        # Load scales and zeros for current group
        group_idx = k_idx // groupsize
        scale = tl.load(scales_ptr + group_idx * stride_scales_g + offs_bn * stride_scales_n)
        zero = tl.load(zeros_ptr + group_idx * stride_zeros_g + offs_bn * stride_zeros_n)
        
        # Load and dequantize B matrix
        b_ptrs = b_ptr + k_idx * stride_bk + offs_bn * stride_bn
        b_vals = tl.load(b_ptrs)
        b_vals = (b_vals.to(tl.float32) - zero) * scale
        
        # Load A matrix
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + (k_idx + offs_k[None, :]) * stride_ak
        a_vals = tl.load(a_ptrs)
        
        # Compute matrix multiplication
        acc += tl.dot(a_vals, b_vals)
    
    # Store result
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(c_ptrs, acc)

@triton.jit
def dequantize_kernel(
    b_ptr, b_scale_ptr, b_zp_ptr, fpb_ptr,
    K, N, group_size,
    stride_bk, stride_bn,
    stride_bsk, stride_bsn,
    stride_bzpk, stride_bzpn,
    stride_fpbk, stride_fpbn,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_k = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Load quantized values
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    b_vals = tl.load(b_ptrs)
    
    # Load scales and zero points
    group_idx = offs_k[:, None] // group_size
    scale_ptrs = b_scale_ptr + group_idx * stride_bsk + offs_n[None, :] * stride_bsn
    zp_ptrs = b_zp_ptr + group_idx * stride_bzpk + offs_n[None, :] * stride_bzpn
    
    scales = tl.load(scale_ptrs)
    zeros = tl.load(zp_ptrs)
    
    # Dequantize
    fp_vals = (b_vals.to(tl.float32) - zeros) * scales
    
    # Store result
    fpb_ptrs = fpb_ptr + offs_k[:, None] * stride_fpbk + offs_n[None, :] * stride_fpbn
    tl.store(fpb_ptrs, fp_vals)

# Helper functions
def dequantize_int4(qweight, scales, zeros):
    K, N = qweight.shape
    groupsize = K // scales.shape[0] if scales.shape[0] > 1 else K
    
    # Output tensor
    output = torch.empty((K, N), dtype=torch.float16, device=qweight.device)
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(K, META['BLOCK_SIZE_K']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    dequantize_kernel[grid](
        qweight, scales, zeros, output,
        K, N, groupsize,
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        output.stride(0), output.stride(1),
    )
    
    return output

def matmul_dequantize_int4_s1(a, qweight, scales, zeros):
    # Get dimensions
    M, K = a.shape
    K, N = qweight.shape
    groupsize = K // scales.shape[0] if scales.shape[0] > 1 else K
    
    # Output tensor
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    matmul4_kernel[grid](
        a, qweight, c,
        scales, zeros,
        M, N, K,
        a.stride(0), a.stride(1),
        qweight.stride(0), qweight.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        groupsize, scales.shape[0],
    )
    
    return c

def quantize_int4(weight, groupsize=128):
    K, N = weight.shape
    assert K % groupsize == 0, "K must be divisible by groupsize"
    
    num_groups = K // groupsize
    weight_groups = weight.reshape(num_groups, groupsize, N)
    
    # Calculate scales and zeros
    weight_max = weight_groups.amax(dim=1)
    weight_min = weight_groups.amin(dim=1)
    
    scales = (weight_max - weight_min) / 15
    zeros = weight_min
    
    # Quantize
    weight_scaled = (weight_groups - zeros.unsqueeze(1)) / scales.unsqueeze(1)
    weight_clipped = torch.clamp(torch.round(weight_scaled), 0, 15)
    
    # Pack into int32
    weight_packed = weight_clipped.to(torch.int32)
    weight_packed = weight_packed.reshape(K, N)
    
    return weight_packed, scales, zeros
