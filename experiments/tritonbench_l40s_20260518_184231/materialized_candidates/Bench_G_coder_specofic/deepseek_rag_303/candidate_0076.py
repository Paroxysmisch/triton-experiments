import math
import torch
import triton
import triton.language as tl

# Kernel to quantize int8 per row.
@triton.jit
def quantize_int8_perrow_kernel(
    fpa,
    a,
    as_,
    M,
    K,
    N_BLOCKS: tl.constexpr,
    N_P2: tl.constexpr,
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
):
    i_m = tl.program_id(0)
    i_k = tl.program_id(1)
    _, i_n_block = tl.swizzle2d(N_BLOCKS, num_warps=num_warps, num_stages=num_stages)
    
    abs_max = -float("inf")
    for i_n_block in range(i_n_block, N_BLOCKS, num_warps * num_warps):
        fa = tl.load(fpa + i_m * K + i_k + i_n_block * N_P2, mask=i_k + i_n_block * N_P2 < K, other=float("-inf"))
        abs_max = tl.maximum(tl.fabs(fa), abs_max)
        
    scale = tl.where(abs_max > 0, abs_max / 127, 1)
    
    for i_n_block in range(i_n_block, N_BLOCKS, num_warps * num_stages):
        fa = tl.load(fpa + i_m * K + i_k + i_n_block * N_P2, mask=i_k + i_n_block * N_P2 < K, other=0)
        i8 = tl.libdevice.llrint(127 * (fa / scale))
        tl.store(a + i_m * K + i_k + i_n_block * N_P2, i8, mask=i_k + i_n_block * N_P2 < K)
        
    tl.store(as_ + i_m, scale)

# Function to call the quantize_int8_perrow_kernel Triton kernel.
@triton.autotune(configs=[
                    triton.Config(num_stages=2, num_warps=2),
                    triton.Config(num_stages=2, num_warps=4),
                    triton.Config(num_stages=2, num_warps=8),
                ],
                key=["K", "M"])
@triton.autotune(configs=[
                    triton.Config(num_stages=2, num_warps=32),
                    triton.Config(num_stages=4, num_warps=32),
                ],
                key=["N"])
def quantize_int8_perrow(fpa,
                          a,
                          as_,
                          M,
                          K,
                          BLOCK_SIZE_N: tl.constexpr = 64):
    N_P2 = 2 ** (tl.cdiv(BLOCK_SIZE_N.to(tl.int64), 8))
    N_BLOCKS = tl.cdiv(BLOCK_SIZE_N, N_P2)
    
    grid = (M, K)
    with torch.cuda.device(fpa.device.index):
        quantize_int8_perrow_kernel[grid](
            fpa,
            a,
            as_,
            M,
            K,
            N_BLOCKS=N_BLOCKS,
            N_P2=N_P2)

# Another Triton kernel to perform matrix multiplication.
@triton.jit
def matmul_kernel(
    a,
    as_ptr,
    b,
    bs_ptr,
    c,
    M,
    N,
    K,
    N_BLOCKS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUPS_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    i_m, i_n, i_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    for i_m_new in range(i_m, M, GROUPS_M):
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.bfloat16)
        
        k_range = range(i_k * SPLIT_K, min((i_k + 1) * SPLIT_K, K))
        prt = as_ptr + i_m_new * N_BLOCKS + tl.arange(i_n, i_n + N_BLOCKS)[:, None]
        abs_max = tl.load(prt, mask=(i_n + N_BLOCKS) <= N, other=0)
        fa = tl.load(a + i_m_new * K + k_range[:, None], mask=(k_range[None, :] < K)[:, :, None], other=0)
        fb = tl.load(b + k_range[:, None] * N + i_n + tl.arange(BLOCK_SIZE_N)[None, :],
                    mask=(k_range[None, :] < K)[:, :, None], other=0)
        accumulator += fa * fb
        
        mask = ((k_range[:, None] + 1)[:, :, None] <= K)[:, :, 0]
        b_new = b + k_range[:, None] * N + tl.arange(BLOCK_SIZE_N)[None, :]
        sa = tl.load(as_ptr + i_m_new * N_BLOCKS + tl.arange(i_n, i_n + N_BLOCKS)[None, :], mask=(i_n + N_BLOCKS) <= N)
        sb = tl.load(bs_ptr + b_new, mask=mask, other=0.0)
        accumulator = (accumulator / sa[:, :, None]) * sb[:, None, :]
        
        accumulator = tl.sum(accumulator, axis=1).to(c.dtype.element_ty) * 2
        tl.store(c + i_m_new * N + i_n + tl.arange(BLOCK_SIZE_N), accumulator, mask=(i_n + N) <= N)

# A function to call the matmul_quantize_int8 kernel.
def matmul_quantize_int8(
    fpa,
    B,
    fC,
    retobj=None,
    split_k=32,
    atol=1e-07,
    rtol=0,
    stream=None,
    out=None,
):
    if fpa.dtype == torch.float16:
        dtype = torch.float32
    else:
        dtype = torch.float16
    device = fpa.device
    M, K = fpa.shape
    N, K = B.shape
    scale_factor = torch.max(torch.max(torch.abs(fpa), dim=1, keepdim=True)[0], torch.abs(B))

    with torch.cuda.device(fpa.device):
        scale_fpa = fpa / scale_factor
        scale_B = B / scale_factor

        fpa = scale_fpa.to(dtype)
        B = scale_B.to(dtype)

        fpa_w = torch.empty(M, K, device=device, dtype=torch.int8)
        scale_w = torch.empty(M, device=device, dtype=torch.float16)
        B_w = torch.empty(K, N, device=device, dtype=torch.int8
