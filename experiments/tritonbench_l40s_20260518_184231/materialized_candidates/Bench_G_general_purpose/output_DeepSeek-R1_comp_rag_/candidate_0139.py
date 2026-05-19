import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel_persistent(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride elements
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # Other parameters
    allow_tf32: tl.constexpr,
):
    # Kernel implementation
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)

    total_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    total_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_blocks = total_blocks_m * total_blocks_n

    for i in range(pid, total_blocks, num_pid):
        block_m = i // total_blocks_n
        block_n = i % total_blocks_n

        offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_SIZE_K):
            a_mask = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)

            b_mask = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)

            accumulator += tl.dot(a, b, allow_tf32=allow_tf32)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        c = accumulator.to(tl.float32)

        offs_cm = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

def matmul_persistent(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Validate inputs
    assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D tensors"
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Dimension mismatch: {K} vs {K_}"
    assert a.dtype == b.dtype, "Inputs must have the same data type"

    # Make sure inputs are contiguous
    a = a.contiguous()
    b = b.contiguous()

    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Determine kernel parameters based on data type
    if a.dtype == torch.float16:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 32
        num_warps = 4
        num_stages = 3
        allow_tf32 = False
    elif a.dtype == torch.float32:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32
        num_warps = 4
        num_stages = 3
        allow_tf32 = True
    else:
        raise ValueError(f"Unsupported data type: {a.dtype}")

    # Compute total number of blocks
    total_blocks_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    total_blocks_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_blocks = total_blocks_m * total_blocks_n

    # Get number of SMs
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count

    # Set grid size to utilize all SMs with persistent threads
    grid = min(num_sms * 4, total_blocks)

    # Launch kernel
    matmul_kernel_persistent[(grid,)](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        allow_tf32=allow_tf32,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return c
