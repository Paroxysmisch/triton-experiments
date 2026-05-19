import torch
import triton
import triton.language as tl

@triton.jit
def dequantize_kernel(
    b, b_scale, fpb,
    K, N,
    stride_bk, stride_b,
    stride_fpbk, stride_fpb,
    GROUP_SIZE_N: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Dequantize kernel.
    Dequantize int8 matrix 'b' and its scale factors 'b_scale' to float matrix 'fpb'.
    """
    # Load block index
    k_group_idx = tl.program_id(0)
    n_group_idx = tl.program_id(1)

    # Compute offsets for block
    offs_k = k_group_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = n_group_idx * BLOCK_SIZE_N * GROUP_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Load segments of 'b' and 'b_scale'
    b_int = tl.load(b + offs_k[None, :] * stride_bk)
    b_scale_int = tl.load(b_scale + offs_k[None, :] * stride_bk)

    # Dequantize
    b_fp = b_int.to(tl.float32) * b_scale_int.to(tl.float32)

    # Write-back output
    tl.store(fpb + offs_k[:, None] * stride_fpbk + offs_n[None, :] * stride_fpb, b_fp)


def matmul_dequantize_int8(a, b, b_scale, left_shift):
    """
    Perform matmul on int8 tensors.
    Dequantize int8 tensor 'b' to float tensor, then perform matmul with input 'a'.
    """
    assert a.is_contiguous(), "Input tensor a must be contiguous"
    assert b.is_contiguous(), "Input tensor b must be contiguous"
    assert b_scale.is_contiguous(), "Input tensor b_scale must be contiguous"

    # Initialize output tensor
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float16)

    # Ensure compatibility of dimensions
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"

    # Constants
    K = a.shape[1]
    N = b.shape[1]
    GROUP_SIZE_N = 128

    # Autotune parameters
    BLOCK_SIZE_N = triton.next_power_of_2(N // GROUP_SIZE_N)
    BLOCK_SIZE_K = triton.next_power_of_2(K)

    # Determine grid
    grid = lambda meta: (
        triton.cdiv(K, meta["BLOCK_SIZE_K"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"] * meta["GROUP_SIZE_N"]),
    )

    # Call Triton kernel
    dequantize_kernel[grid](
        b,
        b_scale,
        c,
        K,
        N,
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        GROUP_SIZE_N=GROUP_SIZE_N,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    # Dot product
    c = torch.mm(a, c)

    # Dequantize
    c = (c.to(torch.float32) / (1 << left_shift)).to(b.dtype)

    return c
