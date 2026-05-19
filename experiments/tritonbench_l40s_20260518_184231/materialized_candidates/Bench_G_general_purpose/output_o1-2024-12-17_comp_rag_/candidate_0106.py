import torch
import triton
import triton.language as tl

@triton.jit
def iv_dependent_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    TYPE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Calculate 'pid_m' and 'pid_n' - distribute work for M / N dimensions
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Compute memory offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create pointers to A and B
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator
    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Double loop (K-dimension blocking)
    k_remaining = K
    loop_idx = 0
    while k_remaining > 0:
        k_block = tl.minimum(BLOCK_SIZE_K, k_remaining)
        # Conditional logic for loading strategy
        if TYPE == "preload":
            # Preloading strategy
            a_block = tl.load(a_ptrs, mask=offs_k[None, :] < k_block, other=0.0)
            b_block = tl.load(b_ptrs, mask=offs_k[:, None] < k_block, other=0.0)
        elif TYPE == "reversed":
            # Some reversed or alternative load
            r_offs_k = BLOCK_SIZE_K - 1 - offs_k
            a_block = tl.load(a_ptr + (offs_m[:, None] * stride_am + r_offs_k[None, :] * stride_ak),
                              mask=r_offs_k[None, :] < k_block, other=0.0)
            b_block = tl.load(b_ptr + (r_offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn),
                              mask=r_offs_k[:, None] < k_block, other=0.0)
        else:
            # Default loading
            a_block = tl.load(a_ptrs, mask=offs_k[None, :] < k_block, other=0.0)
            b_block = tl.load(b_ptrs, mask=offs_k[:, None] < k_block, other=0.0)

        accum += tl.dot(a_block, b_block)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        k_remaining -= BLOCK_SIZE_K
        loop_idx += 1

    # Convert accum to half and store
    c_data = accum.to(tl.float16)
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    c_mask = mask_m & mask_n
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c_data, mask=c_mask)


def iv_dependent_matmul_wrapper(M, N, K, type_str="default", device='cuda'):
    # Prepare random input tensors
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)
    triton_output = torch.empty((M, N), device=device, dtype=torch.float16)

    # Grid configuration
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    # Set pipeline stages / warps based on type_str (simple illustration)
    if type_str == "preload":
        num_stages = 4
        num_warps = 4
    elif type_str == "reversed":
        num_stages = 3
        num_warps = 4
    else:
        num_stages = 2
        num_warps = 2

    iv_dependent_matmul_kernel[grid](
        a, b, triton_output,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        triton_output.stride(0), triton_output.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
        TYPE=type_str,
        num_stages=num_stages,
        num_warps=num_warps
    )

    return triton_output
