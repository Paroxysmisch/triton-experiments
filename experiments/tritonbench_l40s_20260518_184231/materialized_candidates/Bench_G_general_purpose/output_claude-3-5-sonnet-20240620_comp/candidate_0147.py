import triton
import triton.language as tl
import torch

@triton.jit
def matmul_tma_load_store(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    OUTPUT_F16: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Calculate current block indices
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create block pointers for the matrices
    # Matrix A: [M, K]
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )
    
    # Matrix B: [K, N]
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1)
    )
    
    # Matrix C: [M, N]
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate through K dimension
    for k in range(0, K, BLOCK_K):
        # Load blocks from A and B
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        
        # Compute matrix multiplication
        acc += tl.dot(a, b)
        
        # Advance block pointers
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    
    # Handle output conversion if needed
    if OUTPUT_F16:
        acc = acc.to(tl.float16)
    
    # Store the result
    tl.store(c_block_ptr, acc)

def wrapper_tma_load_store(M, N, K, num_warps=4, num_ctas=1, 
                          transpose_a=False, transpose_b=False, 
                          output_f16=False):
    # Create random input tensors
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    
    # Handle transpositions if needed
    if transpose_a:
        a = a.transpose(0, 1).contiguous()
    if transpose_b:
        b = b.transpose(0, 1).contiguous()
    
    # Allocate output tensor
    c = torch.empty((M, N), device='cuda', 
                   dtype=torch.float16 if output_f16 else torch.float32)
    
    # Calculate grid size
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    # Define block sizes
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    
    # Launch kernel
    matmul_tma_load_store[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        OUTPUT_F16=output_f16,
        num_warps=num_warps,
        num_ctas=num_ctas
    )
    
    return c
