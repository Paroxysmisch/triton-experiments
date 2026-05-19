import torch
import triton
import triton.language as tl

@triton.jit
def matmul4_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Pointers to scales and zeros
    scales_ptr, zeros_ptr,
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
    GROUP_SIZE_M: tl.constexpr
):
    """
    Kernel for computing the matrix multiplication C = A x B
    where B is quantized to 4 bits
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Load scales and zeros
    scales = tl.load(scales_ptr + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    zeros = tl.load(zeros_ptr + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load and dequantize B
        b_idx = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        b_mask = b_idx < K
        b = tl.load(b_ptrs, mask=b_mask[:, None])
        
        # Extract 4-bit values and dequantize
        b = (b >> (tl.arange(0, 8)[:, None] * 4)) & 0xF
        b = b.to(tl.float32)
        b = b * scales[None, :] - zeros[None, :]
        
        # Load A
        a = tl.load(a_ptrs, mask=b_mask[:, None])
        
        # Compute matrix multiplication
        accumulator += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store results
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def matmul_dequantize_int4_gptq(x, qweight, scales, qzeros, output=None):
    """
    Wrapper function for the 4-bit quantized matrix multiplication kernel
    """
    # Handle dimensions
    M, K = x.shape
    N = qweight.shape[1]
    
    # Initialize output if not provided
    if output is None:
        output = torch.empty((M, N), device=x.device, dtype=torch.float16)

    # Configure meta-parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    
    # Create grid
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    # Launch kernel
    matmul4_kernel[grid](
        x, qweight, output,
        scales, qzeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return output

@triton.jit
def dequantize_kernel(
    b_ptr, output_ptr,
    scales_ptr, zeros_ptr,
    K, N,
    stride_bk, stride_bn,
    stride_ok, stride_on,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    """
    Kernel for dequantizing 4-bit matrix to fp16
    """
    pid = tl.program_id(0)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    pid_k = pid // num_pid_k
    pid_n = pid % num_pid_k

    # Block pointers
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Load scales and zeros
    scales = tl.load(scales_ptr + offs_n)
    zeros = tl.load(zeros_ptr + offs_n)
    
    # Load quantized values
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    b = tl.load(b_ptrs)
    
    # Dequantize
    b = (b >> (tl.arange(0, 8)[:, None] * 4)) & 0xF
    b = b.to(tl.float32)
    b = b * scales[None, :] - zeros[None, :]
    
    # Store result
    output_ptrs = output_ptr + offs_k[:, None] * stride_ok + offs_n[None, :] * stride_on
    mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    tl.store(output_ptrs, b, mask=mask)

def dequantize_int4(qweight, scales, qzeros):
    """
    Wrapper function for dequantization kernel
    """
    K, N = qweight.shape
    output = torch.empty((K, N), device=qweight.device, dtype=torch.float16)
    
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_N = 32
    
    grid = (triton.cdiv(K, BLOCK_SIZE_K) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    dequantize_kernel[grid](
        qweight, output,
        scales, qzeros,
        K, N,
        qweight.stride(0), qweight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output
