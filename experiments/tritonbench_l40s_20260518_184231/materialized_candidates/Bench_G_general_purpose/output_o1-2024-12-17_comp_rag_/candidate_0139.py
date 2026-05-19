import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel_persistent(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_tiles_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_tiles_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_tiles = num_tiles_m * num_tiles_n

    tile_idx = pid
    while tile_idx < total_tiles:
        tm = tile_idx // num_tiles_n
        tn = tile_idx % num_tiles_n
        m_off = tm * BLOCK_SIZE_M
        n_off = tn * BLOCK_SIZE_N

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        k_off = 0
        while k_off < K:
            a_offset = (
                (m_off + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_am
                + (k_off + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_ak
            )
            b_offset = (
                (k_off + tl.arange(0, BLOCK_SIZE_K))[:, None] * stride_bk
                + (n_off + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_bn
            )
            a_mask = (
                (m_off + tl.arange(0, BLOCK_SIZE_M))[:, None] < M
            ) & (
                (k_off + tl.arange(0, BLOCK_SIZE_K))[None, :] < K
            )
            b_mask = (
                (k_off + tl.arange(0, BLOCK_SIZE_K))[:, None] < K
            ) & (
                (n_off + tl.arange(0, BLOCK_SIZE_N))[None, :] < N
            )
            a_block = tl.load(a_ptr + a_offset, mask=a_mask, other=0.0)
            b_block = tl.load(b_ptr + b_offset, mask=b_mask, other=0.0)
            acc += tl.dot(a_block, b_block)
            k_off += BLOCK_SIZE_K

        c_offset = (
            (m_off + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_cm
            + (n_off + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_cn
        )
        c_mask = (
            (m_off + tl.arange(0, BLOCK_SIZE_M))[:, None] < M
        ) & (
            (n_off + tl.arange(0, BLOCK_SIZE_N))[None, :] < N
        )
        tl.store(c_ptr + c_offset, acc, mask=c_mask)
        tile_idx += GROUP_SIZE

def matmul_persistent(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, "Inner dimensions must match"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    device_props = torch.cuda.get_device_properties(a.device)
    NUM_SMS = device_props.multi_processor_count

    grid = lambda META: (NUM_SMS,)
    matmul_kernel_persistent[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE=NUM_SMS
    )
    return c
