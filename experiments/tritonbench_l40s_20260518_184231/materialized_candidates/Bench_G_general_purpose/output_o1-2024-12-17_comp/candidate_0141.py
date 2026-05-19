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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_blk_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blk_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_blocks = num_blk_m * num_blk_n

    offset = pid
    while offset < total_blocks:
        block_m = offset // num_blk_n
        block_n = offset % num_blk_n

        row_offsets = tl.arange(0, BLOCK_SIZE_M)
        col_offsets = tl.arange(0, BLOCK_SIZE_N)
        m_tile = block_m * BLOCK_SIZE_M + row_offsets
        n_tile = block_n * BLOCK_SIZE_N + col_offsets

        # Create accumulator in fp32
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        k0 = 0
        while k0 < K:
            k_offsets = tl.arange(0, BLOCK_SIZE_K)
            kk = k0 + k_offsets

            a_mask = (m_tile[:, None] < M) & (kk[None, :] < K)
            b_mask = (kk[:, None] < K) & (n_tile[None, :] < N)

            a_ptrs = a_ptr + (m_tile[:, None] * stride_am) + (kk[None, :] * stride_ak)
            b_ptrs = b_ptr + (kk[:, None] * stride_bk) + (n_tile[None, :] * stride_bn)

            a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)

            accumulator += tl.dot(a_vals, b_vals)
            k0 += BLOCK_SIZE_K

        c_mask = (m_tile[:, None] < M) & (n_tile[None, :] < N)
        # Convert accumulator to the same dtype as output
        c_vals = accumulator
        c_ptrs = c_ptr + (m_tile[:, None] * stride_cm) + (n_tile[None, :] * stride_cn)
        tl.store(c_ptrs, c_vals, mask=c_mask)

        offset += num_programs


def matmul_persistent(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2, "Matrices must be 2D"
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Incompatible matrix dimensions"
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Get strides
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    # Get number of SMs to set the persistent grid size
    device_id = a.device.index
    num_sms = torch.cuda.get_device_properties(device_id).multi_processor_count

    # Kernel launch
    matmul_kernel_persistent[
        num_sms
    ](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32
    )
    return c
