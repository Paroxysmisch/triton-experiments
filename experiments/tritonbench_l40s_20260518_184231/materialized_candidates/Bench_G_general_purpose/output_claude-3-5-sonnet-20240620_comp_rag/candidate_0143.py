import torch
import triton
import triton.language as tl

@triton.jit
def matmul_tma_load_store(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    OUTPUT_F16: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute block indices
    block_m = pid // (N // BLOCK_N)
    block_n = pid % (N // BLOCK_N)
    
    # Compute offsets for this block
    offs_m = block_m * BLOCK_M
    offs_n = block_n * BLOCK_N
    
    # Create block pointers with proper offsets
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(offs_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )
    
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, offs_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1)
    )
    
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(offs_m, offs_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate over K dimension
    for k in range(0, K, BLOCK_K):
        # Load blocks from A and B
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        
        # Compute matrix multiplication for this block
        acc += tl.dot(a, b)
        
        # Advance block pointers
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    
    # Convert output to float16 if requested
    if OUTPUT_F16:
        acc = acc.to(tl.float16)
    
    # Store the result
    tl.store(c_block_ptr, acc)

def wrapper_tma_load_store(M, N, K, NUM_WARPS=4, NUM_CTAS=1, 
                         TRANS_A=False, TRANS_B=False, OUTPUT_F16=False,
                         BLOCK_M=16, BLOCK_N=16, BLOCK_K=16):
    # Prepare input matrices
    if TRANS_A:
        a = torch.randn((K, M), device='cuda', dtype=torch.float16).T
    else:
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        
    if TRANS_B:
        b = torch.randn((N, K), device='cuda', dtype=torch.float16).T
    else:
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    
    # Prepare output matrix with appropriate dtype
    c = torch.empty((M, N), device='cuda',
                   dtype=torch.float16 if OUTPUT_F16 else torch.float32)
    
    # Calculate grid size
    grid = (M * N) // (BLOCK_M * BLOCK_N)
    
    # Launch kernel
    matmul_tma_load_store[(grid,)](
        a_ptr=a, b_ptr=b, c_ptr=c,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS,
        num_ctas=NUM_CTAS,
        OUTPUT_F16=OUTPUT_F16
    )
    
    return c
