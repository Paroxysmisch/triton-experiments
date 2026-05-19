import torch
import triton
import triton.language as tl

@triton.jit
def matmul4_kernel(
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
    # Pointer to scale and zero points for quantization
    scale_ptr, zero_ptr,
    groupsize: tl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B
    A is of shape (M, K) float16
    B is of shape (K, N) int32 containing 4-bit values
    C is of shape (M, N) float16
    """
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    
    # Get the block ID
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # The memory addresses of elements in the first block of A and B that we'll load
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize the accumulator to zero
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate to compute a block of the C matrix
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the scale and zero point for this block
        scale_idx = k * BLOCK_SIZE_K // groupsize
        scale = tl.load(scale_ptr + scale_idx)
        zero = tl.load(zero_ptr + scale_idx)
        
        # Load a block of A
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + (k * BLOCK_SIZE_K + offs_k[None, :]) * stride_ak,
                   mask=(offs_am[:, None] < M) & (k * BLOCK_SIZE_K + offs_k[None, :] < K))
        
        # Load a block of B and dequantize
        b_idx = k * BLOCK_SIZE_K + offs_k[:, None]
        b_packed = tl.load(b_ptr + b_idx * stride_bk + offs_bn[None, :] * stride_bn,
                          mask=(b_idx < K) & (offs_bn[None, :] < N))
        
        # Unpack 4-bit values
        b_unpacked = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
        for i in range(8):  # 8 values per 32-bit word
            mask = 0xF << (i * 4)
            values = (b_packed & mask) >> (i * 4)
            b_unpacked = b_unpacked + (values.to(tl.float32) - zero) * scale
        
        # Compute the matrix multiplication
        acc += tl.dot(a.to(tl.float32), b_unpacked)
    
    # Store the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c = acc.to(tl.float16)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
             c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

# The wrapper function
def matmul_dequantize_int4_gptq(x, qweight, scales, qzeros, output=None):
    """
    Compute the matrix multiplication of x and quantized weights.
    Args:
        x: Input tensor (M, K)
        qweight: Quantized weight tensor (K, N) packed in int32
        scales: Scale factors for dequantization
        qzeros: Zero points for dequantization
        output: Optional output tensor
    Returns:
        Output tensor (M, N)
    """
    M, K = x.shape
    N = qweight.shape[1]
    
    # Allocate output if not provided
    if output is None:
        output = torch.empty((M, N), device=x.device, dtype=torch.float16)
    
    # Configure meta-parameters
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    GROUP_SIZE = 128  # Typical group size for GPTQ quantization
    
    # Create launch grid
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    # Launch kernel
    matmul4_kernel[grid](
        x, qweight, output,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        scales, qzeros,
        GROUP_SIZE,
    )
    
    return output
