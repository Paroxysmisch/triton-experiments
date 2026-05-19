import torch
import triton
import triton.language as tl

@triton.jit
def quantize_int8_perrow_kernel(
    fpa_ptr, a_ptr, as_ptr,
    M, K,
    stride_m_fpa, stride_k_fpa,
    stride_m_a, stride_k_a,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Calculate row index
    row_idx = pid * BLOCK_SIZE_M
    
    # Initialize max values for the row
    max_val = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Load and find max values
    for k in range(0, K, BLOCK_SIZE_K):
        k_mask = k + tl.arange(0, BLOCK_SIZE_K) < K
        for m in range(BLOCK_SIZE_M):
            m_idx = row_idx + m
            if m_idx < M:
                vals = tl.load(fpa_ptr + m_idx * stride_m_fpa + k * stride_k_fpa, 
                             mask=k_mask, other=0.0)
                max_val[m] = tl.maximum(max_val[m], tl.abs(vals).max())
    
    # Calculate scales
    scales = max_val / 127.0
    
    # Store scales
    for m in range(BLOCK_SIZE_M):
        m_idx = row_idx + m
        if m_idx < M:
            tl.store(as_ptr + m_idx, scales[m])
    
    # Quantize and store values
    for k in range(0, K, BLOCK_SIZE_K):
        k_mask = k + tl.arange(0, BLOCK_SIZE_K) < K
        for m in range(BLOCK_SIZE_M):
            m_idx = row_idx + m
            if m_idx < M:
                vals = tl.load(fpa_ptr + m_idx * stride_m_fpa + k * stride_k_fpa,
                             mask=k_mask, other=0.0)
                quant_vals = tl.math.round(vals / scales[m])
                tl.store(a_ptr + m_idx * stride_m_a + k * stride_k_a,
                        quant_vals.to(tl.int8), mask=k_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    as_ptr, bs_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = 1
    
    # Calculate indices
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Block pointers
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Load scales
    a_scale = tl.load(as_ptr + offs_am)
    b_scale = tl.load(bs_ptr + offs_bn)
    
    # Main loop
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        b = tl.load(b_ptr + (k + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn)
        
        # Convert to float and apply scales
        a = a.to(tl.float32) * a_scale[:, None]
        b = b.to(tl.float32) * b_scale[None, :]
        
        # Accumulate
        accumulator += tl.dot(a, b)
    
    # Store results
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             accumulator, mask=mask)

def quantize_int8_perrow(fpa):
    M, K = fpa.shape
    
    # Allocate output tensors
    a = torch.empty_like(fpa, dtype=torch.int8)
    as_tensor = torch.empty((M,), dtype=torch.float32, device=fpa.device)
    
    # Launch kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),)
    quantize_int8_perrow_kernel[grid](
        fpa_ptr=fpa, a_ptr=a, as_ptr=as_tensor,
        M=M, K=K,
        stride_m_fpa=K, stride_k_fpa=1,
        stride_m_a=K, stride_k_a=1,
        BLOCK_SIZE_M=32, BLOCK_SIZE_K=128
    )
    
    return a, as_tensor

def matmul_int8(a, b, a_scale, b_scale, out=None):
    M, K = a.shape
    _, N = b.shape
    
    # Allocate output if needed
    if out is None:
        out = torch.empty((M, N), dtype=torch.float32, device=a.device)
    
    # Launch kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=out,
        as_ptr=a_scale, bs_ptr=b_scale,
        M=M, N=N, K=K,
        stride_am=K, stride_ak=1,
        stride_bk=N, stride_bn=1,
        stride_cm=N, stride_cn=1,
    )
    
    return out

def matmul_quantize_int8(fpa, fpb):
    # Quantize inputs
    a, a_scale = quantize_int8_perrow(fpa)
    b, b_scale = quantize_int8_perrow(fpb)
    
    # Perform quantized matmul
    return matmul_int8(a, b, a_scale, b_scale)
