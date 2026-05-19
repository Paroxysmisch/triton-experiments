import triton
import triton.language as tl

@triton.jit
def _quantize_global_transpose(
    A, B,
    stride_am, stride_an,
    stride_bm, stride_bn,
    absmax_inv,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid_m_total = tl.program_id(0)
    pid_n = tl.program_id(1)

    group_size_m = (M + BLOCK_M * GROUP_M - 1) // (BLOCK_M * GROUP_M)
    pid_m = pid_m_total % group_size_m
    group_idx_m = pid_m_total // group_size_m

    off_m = group_idx_m * GROUP_M * BLOCK_M + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (off_m[:, None] < M) & (off_n[None, :] < N)
    a = tl.load(
        A + off_m[:, None] * stride_am + off_n[None, :] * stride_an,
        mask=mask,
        other=0.0
    )

    quant = tl.cast(tl.round(a * absmax_inv), tl.int8)

    mask_t = (off_n[:, None] < N) & (off_m[None, :] < M)
    tl.store(
        B + off_n[:, None] * stride_bm + off_m[None, :] * stride_bn,
        quant,
        mask=mask_t
    )

def quantize_global_transpose(A, B, stride_am, stride_an, stride_bm, stride_bn, M, N, GROUP_M=1, BLOCK_M=128, BLOCK_N=128):
    import torch
    absmax = torch.max(torch.abs(A))
    absmax_inv = 127.0 / absmax
    B.zero_()

    def grid(meta):
        group_size_m = (M + meta['BLOCK_M'] * meta['GROUP_M'] - 1) // (meta['BLOCK_M'] * meta['GROUP_M'])
        grid_m = group_size_m * meta['GROUP_M']
        grid_n = (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N']
        return (grid_m, grid_n)

    _quantize_global_transpose[grid](
        A, B,
        stride_am, stride_an,
        stride_bm, stride_bn,
        absmax_inv,
        M, N,
        BLOCK_M, BLOCK_N,
        GROUP_M
    )
