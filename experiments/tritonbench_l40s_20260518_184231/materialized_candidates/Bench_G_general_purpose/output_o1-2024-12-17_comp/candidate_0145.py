import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}),
        triton.Config({"BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}),
    ],
    key=["K", "N"],
)
@triton.jit
def dequantize_kernel(
    b_ptr,             # *int8
    b_scale_ptr,       # *float32
    fpb_ptr,           # *float32
    K,                 # int
    N,                 # int
    stride_bk,         # int
    stride_bn,         # int
    stride_b_scale,    # int
    stride_fpbk,       # int
    stride_fpbn,       # int
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Offsets for the block
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # Create the 2D indices
    bnk = (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)
    scale_idx = offs_n * stride_b_scale
    fpb_idx = (offs_k[:, None] * stride_fpbk) + (offs_n[None, :] * stride_fpbn)

    # Masks
    mask_k = offs_k < K
    mask_n = offs_n < N

    # Load int8 b
    b_vals = tl.load(b_ptr + bnk, mask=(mask_k[:, None] & mask_n[None, :]), other=0)
    b_vals_f32 = b_vals.to(tl.float32)

    # Load scales
    b_scale = tl.load(b_scale_ptr + scale_idx, mask=mask_n, other=0)
    b_scale = b_scale[None, :]

    # Dequantize
    fpb_vals = b_vals_f32 * b_scale

    # Store results
    tl.store(fpb_ptr + fpb_idx, fpb_vals, mask=(mask_k[:, None] & mask_n[None, :]))


def matmul_dequantize_int8(a: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor):
    """
    Dequantize int8 matrix b using b_scale, then do matrix multiplication: c = a.mm(fpb).
    a:       [M, K] (float32)
    b:       [K, N] (int8)
    b_scale: [N]    (float32) scale factors for columns of b
    Returns: c = [M, N] (float32)
    """
    assert a.shape[1] == b.shape[0], "Shapes of A and B are not compatible for matmul"
    K, N = b.shape

    # Allocate fpb as float32
    fpb = torch.empty_like(b, dtype=torch.float32)

    # Launch Triton kernel to dequantize b
    grid = lambda META: (
        (N + META["BLOCK_SIZE_N"] - 1) // META["BLOCK_SIZE_N"],
        (K + META["BLOCK_SIZE_K"] - 1) // META["BLOCK_SIZE_K"]
    )

    dequantize_kernel[grid](
        b, b_scale, fpb,
        K, N,
        b.stride(0), b.stride(1),
        b_scale.stride(0),
        fpb.stride(0), fpb.stride(1)
    )

    # Perform matrix multiplication
    c = a.mm(fpb)
    return c
