import torch
import triton
import triton.language as tl

# Triton kernel with conditional logic for different loading strategies
@triton.jit
def iv_dependent_matmul_kernel(
    a, b, c,  # Pointers to matrices
    M, N, K,  # Matrix dimensions
    stride_am, stride_ak,  # Strides for matrix a
    stride_bk, stride_bn,  # Strides for matrix b
    stride_cm, stride_cn,  # Strides for matrix c
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    type: tl.constexpr,  # Type of preloading to use
):
    """
    Kernel for computing the matmul C = A x B
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    # m offsets of the blocks
    m_offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # n offsets of the blocks
    n_offsets = tl.arange(0, BLOCK_SIZE_N)

    # Define a range of k values
    k_range = tl.arange(0, BLOCK_SIZE_K)

    # pointers
    a_ptrs = a + m_offsets[:, None] * stride_am + k_range[None, :] * stride_ak
    b_ptrs = b + k_range[:, None] * stride_bk + n_offsets[None, :] * stride_bn
    c_ptrs = c + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn

    # Create a mask
    a_mask = (m_offsets[:, None] < M) & (k_range[None, :] < K)
    b_mask = (k_range[:, None] < K) & (n_offsets[None, :] < N)
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if type == "A":
        tiled_a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        for i in range(K // BLOCK_SIZE_K):
            b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(tiled_a, b_block, allow_tf32=True)
            b_ptrs += BLOCK_SIZE_K * stride_bk
            b_mask = (k_range[:, None] + i * BLOCK_SIZE_K < K) & b_mask
    elif type == "B":
        tiled_b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        for i in range(K // BLOCK_SIZE_K):
            a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)
            acc += tl.dot(a_block, tiled_b, allow_tf32=True)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            a_mask = (k_range[None, :] + i * BLOCK_SIZE_K < K) & a_mask
    elif type == "C":
        for i in range(K // BLOCK_SIZE_K):
            tiled_a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(tiled_a, b_block, allow_tf32=True)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
            a_mask = (k_range[None, :] + i * BLOCK_SIZE_K < K) & a_mask
            b_mask = (k_range[:, None] + i * BLOCK_SIZE_K < K) & b_mask
    else:
        tiled_a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        tiled_b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(tiled_a, tiled_b, allow_tf32=True)

    c = tl.load(c_ptrs, mask=c_mask, other=0.0)
    acc += c
    tl.store(c_ptrs, acc.to(c.dtype.element_ty), mask=c_mask)

# Wrapper function to invoke the Triton kernel
def iv_dependent_matmul_wrapper(M, N, K, type, device, dtype, pin_memory, scheduling_type):
    if device == "random":
        device = torch.cuda.device(torch.randint(torch.cuda.device_count()).item())
    else:
        device = torch.cuda.device(int(device))

    torch.cuda.device(device)

    a = torch.randn((M, K), device="cuda", dtype=dtype, pin_memory=pin_memory)
    b = torch.randn((K, N), device="cuda", dtype=dtype, pin_memory=pin_memory)
    c = torch.randn((M, N), device="cuda", dtype=dtype, pin_memory=pin_memory)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),)

    if scheduling_type == "grid":
        num_stages = 3
    elif scheduling_type == "dynamic":
        num_stages = 1
    else:
        raise NotImplementedError

    iv_dependent_matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
        type=type,
        num_stages=num_stages,
    )
    triton_output = c.to("cpu").clone()
    return triton_output
