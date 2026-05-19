import torch
import triton
import triton.language as tl

@triton.jit
def matmul_tma_load_store(
    # Matrix pointers
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride definitions
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Kernel parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUTPUT_F16: tl.constexpr,
    trans_a: tl.constexpr,
    trans_b: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # TMA grid for K dimension
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    
    for k in range(0, num_k_blocks):
        # Create block pointers for A
        if trans_a:
            a_block_ptr = tl.make_block_ptr(
                base=a_ptr,
                shape=(K, M),
                strides=(stride_ak, stride_am),
                offsets=(k * BLOCK_K, pid_m * BLOCK_M),
                block_shape=(BLOCK_K, BLOCK_M),
                order=(0, 1)
            )
        else:
            a_block_ptr = tl.make_block_ptr(
                base=a_ptr,
                shape=(M, K),
                strides=(stride_am, stride_ak),
                offsets=(pid_m * BLOCK_M, k * BLOCK_K),
                block_shape=(BLOCK_M, BLOCK_K),
                order=(1, 0)
            )
        
        # Create block pointers for B
        if trans_b:
            b_block_ptr = tl.make_block_ptr(
                base=b_ptr,
                shape=(N, K),
                strides=(stride_bn, stride_bk),
                offsets=(pid_n * BLOCK_N, k * BLOCK_K),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(0, 1)
            )
        else:
            b_block_ptr = tl.make_block_ptr(
                base=b_ptr,
                shape=(K, N),
                strides=(stride_bk, stride_bn),
                offsets=(k * BLOCK_K, pid_n * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(1, 0)
            )
        
        # Load blocks
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        
        # Accumulate matrix product
        acc += tl.dot(a, b, out_dtype=tl.float32)
    
    # Create output block pointer
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    
    # Convert and store output
    if OUTPUT_F16:
        acc = acc.to(tl.float16)
    tl.store(c_block_ptr, acc, boundary_check=(0, 1))

def warpper_tma_load_store(
    M, N, K,
    trans_a=False, trans_b=False,
    output_f16=True,
    BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
    num_warps=4, num_ctas=1
):
    # Create input tensors with appropriate shapes
    a_shape = (K, M) if trans_a else (M, K)
    b_shape = (N, K) if trans_b else (K, N)
    
    a = torch.randn(a_shape, device='cuda', dtype=torch.float16)
    b = torch.randn(b_shape, device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', 
                   dtype=torch.float16 if output_f16 else torch.float32)
    
    # Calculate strides accounting for transpositions
    stride_am = a.stride(1) if trans_a else a.stride(0)
    stride_ak = a.stride(0) if trans_a else a.stride(1)
    stride_bk = b.stride(1) if trans_b else b.stride(0)
    stride_bn = b.stride(0) if trans_b else b.stride(1)
    
    # Launch kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matmul_tma_load_store[grid](
        a, b, c, M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        OUTPUT_F16=output_f16,
        trans_a=trans_a,
        trans_b=trans_b,
        num_warps=num_warps,
        num_ctas=num_ctas
    )
    
    return c
