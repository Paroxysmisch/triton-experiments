import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=2),
    ],
    key=["M", "N", "K", "NO_GROUP"],
)
@triton.jit
def matmul4_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scales_ptr,
    zeros_ptr,
    g_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scales_g,
    stride_scales_n,
    stride_scales_k,
    stride_zeros_g,
    stride_zeros_n,
    stride_zeros_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NO_GROUP: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, K) float16
    B is of shape (K//8, N) int32
    C is of shape (M, N) float16
    scales is of shape (G, N) float16
    zeros is of shape (G, N) float16
    g_ptr is of shape (K) int32
    """
    infearure_per_bits = 32

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_mask = offs_am[:, None] < M
    # b_ptrs is set up such that it repeats elements along the K axis 8 times
    b_ptrs = b_ptr + (
        (offs_k[:, None] // infearure_per_bits) * stride_bk
        + offs_bn[None, :] * stride_bn
    )  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    g_ptrs = g_ptr + offs_k
    # shifter is used to extract the N bits of each element in the 32-bit word from B
    scales_ptrs = scales_ptr + offs_bn[None, :]
    zeros_ptrs = zeros_ptr + (offs_bn[None, :] // infearure_per_bits)

    shifter = (offs_k % infearure_per_bits) * 4
    zeros_shifter = (offs_bn % infearure_per_bits) * 4
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, num_pid_k):
        g_idx = tl.load(g_ptrs)

        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(
            scales_ptrs + g_idx[:, None] * stride_scales_g,
            mask=(offs_bn[None, :] < N),
            other=0.0,
        )  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
        zeros = tl.load(
            zeros_ptrs + g_idx[:, None] * stride_zeros_g,
            mask=(offs_bn[None, :] < N),
            other=0.0,
        )  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

        zeros = (zeros >> zeros_shifter[None, :]) & 0xF
        zeros = (zeros + 1)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

        # Now we need to unpack b (which is N-bit values) into 32-bit values
        b = (b >> shifter[:, None]) & 0xF  # Extract the N-bit values
        b = (b - zeros) * scales  # Scale and shift

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
        g_ptrs += BLOCK_SIZE_K

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul_dequantize_int4_gptq(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    g: torch.Tensor,
    use_triton: bool = True,
    inplace: bool = False,
) -> torch.Tensor:
    assert a.shape[1] == b.shape[1]
    assert a.is_contiguous()
    assert g.is_contiguous()

    M, K = a.shape
    K, N = b.shape

    if not inplace:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    if use_triton:
        matmul4_kernel[
            lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
        ](
            a,
            b,
            c,
            scales,
            zeros,
            g,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            scales.stride(0),
            scales.stride(1),
            scales.stride(2),
            zeros.stride(0),
            zeros.stride(1),
            zeros.stride(2),
            NO_GROUP=(g.numel() == 1),
        )
    else:
        b = b.float()
        b = b / (1 << (32 - 4))
        b = b + 1

        zeros = zeros + 1
        c = torch.matmul(a, b)
        c = c * scales
        c = c / zeros

    return c


def quantize_int4(x: torch.Tensor) -> torch.Tensor:
    """
    Quantize matrix x to int4 using GPTQ algorithm.
    The output tensor will be in the shape of (C, K//8), where K is the last dimension of the input tensor.
    """
    # Check constraints
    assert x.is_contiguous()
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32]

    # Get dimensions
    *leading_dims, K = x.shape
    N = 32 * torch.div(K, 32, rounding_mode="ceil").int()
    K = K.int()
    x = x.reshape(-1, K)

    t = torch.transpose(x,
