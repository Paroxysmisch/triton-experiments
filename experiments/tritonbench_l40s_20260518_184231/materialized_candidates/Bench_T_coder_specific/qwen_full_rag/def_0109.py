import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import _volume

# This kernel computes the svd of a matrix
@triton.jit
def _linalg_svd_compute(
    U_ptr,
    S_ptr,
    Vh_ptr,
    A_ptr,
    M,
    N,
    min_m_n,
    max_m_n,
    stride_am,
    stride_an,
    stride_um,
    stride_un,
    stride_vm,
    stride_vn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    # now compute the block that each program will go through
    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Move to this block
    A_block_ptr = A_ptr + offset_m[:, None] * stride_am + offset_n[None, :] * stride_an
    u_block_ptr = U_ptr + offset_m[:, None] * stride_um + offset_n[None, :] * stride_un
    v_block_ptr = (
        Vh_ptr + offset_m[:, None] * stride_vm + offset_n[None, :] * stride_vn
    )

    # Do the computation
    for i in range(0, max_m_n, BLOCK_SIZE_M):
        # Prefetch data
        a = tl.load(
            A_block_ptr,
            mask=(offset_m < M)[:, None] & (offset_n < N)[None, :],
            other=0.0,
        )
        s = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        v = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        # do computation
        if i < min_m_n:
            s += tl.dot(a, tl.trans(a))
        else:
            s += tl.dot(a, tl.trans(a[:, 0:min_m_n]))
        # update s
        s = tl.where(offset_m[:, None] <= min_m_n - 1, tl.sqrt(s), 0.0)
        # # update u
        u = a
        # # normalize u
        u = u / s[:, None]
        tl.store(u_block_ptr, u, mask=(offset_m < M)[:, None])
        # update v
        if i < min_m_n:
            v = tl.trans(tl.solve(tl.trans(a), tl.trans(v)))
        else:
            v = tl.trans(tl.solve(tl.trans(a[:, 0:min_m_n]), tl.trans(v[:, 0:i])))
        tl.store(v_block_ptr, v, mask=(offset_n < N)[None, :])
        # move the block ptrs forward
        A_block_ptr += BLOCK_SIZE_M * stride_am
        u_block_ptr += BLOCK_SIZE_M * stride_um
        v_block_ptr += BLOCK_SIZE_M * stride_vm

# This kernel computes the full uv of svd
@triton.jit
def _linalg_svd_full_uv_compute(
    U_ptr,
    Vh_ptr,
    A_ptr,
    M,
    N,
    stride_am,
    stride_an,
    stride_um,
    stride_un,
    stride_vm,
    stride_vn,
    LDA: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    # now compute the block that each program will go through
    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Move to this block
    A_block_ptr = A_ptr + offset_m[:, None] * stride_am + offset_n[None, :] * stride_an
    u_block_ptr = U_ptr + offset_m[:, None] * stride_um + offset_n[None, :] * stride_un
    v_block_ptr = (
        Vh_ptr + offset_m[:, None] * stride_vm + offset_n[None, :] * stride_vn
    )

    # Do the computation
    for i in range(0, N, BLOCK_SIZE_N):
        # Prefetch data
        a = tl.load(
            A_block_ptr,
            mask=(offset_m < M)[:, None] & (offset_n < N)[None, :],
            other=0.0,
        )
        # print(a)
        s = tl.zeros((LDA, BLOCK_SIZE_N), dtype=tl.float32)
        v = tl.zeros((LDA, BLOCK_SIZE_N), dtype=tl.float32)
        # do computation
        s += tl.dot(a, tl.trans(a))
        # print(s)
        s = tl.sqrt(tl.diag(s))
        # update u
        u = a / s[:, None]
        # print(u)
        tl.store(u_block_ptr, u, mask=(offset_m < M)[:, None])
        # update v
        v = tl.trans(tl.solve(tl.trans(a), tl.trans(v)))
        # print(v)
        tl.store(v_block_ptr, v, mask=(offset_n < N)[None, :])
        # move the block ptrs forward
        A_block_ptr += BLOCK_SIZE_N * stride_an
        u_block_ptr += BLOCK_SIZE_N * stride_un
        v_block_ptr += BLOCK_SIZE_N * stride_vn

# Configs for Zeros like
class _ZEROS_LIKE_CONFIGS:
    ENABLE = True
    DTYPE = "ALL"
    CONTIGUOUS = [True, False]
    STRIDE_PATTERNS = ["DEFAULT", "ZERO_STRIDED"]
    BLOCK_SIZE_M = [1, 2, 4]
    BLOCK_SIZE_N = [1024, 2048, 4096]

# Test linalg svd
def test_linalg_svd(M, N, BATCH, DTYPES, FULL_MATRICES, RCONDS, TRANSPOSINGS, DEV_IDS):
    REPEAT = 100
    rtol = 1e-2
    atol = 1e-3
    check = "svd_check"
    for _ in range(REPEAT):
        for batch in BATCH:
            for dtype_str in DTYPES:
                for transposing in TRANSPOSINGS:
                    for full_matrix in FULL_MATRICES:
                        for rcond in RCONDS:
                            for dev_id in DEV_IDS:
                                # Skip half precision testing for small matrices
                                if (
                                    str(dtype_str) == "torch.float16"
                                    and M * N < 2048 * 2048
                                ):
                                    continue
                                # Create a fresh, randomly initialized CPU array
                                a_cpu = (
                                    torch.randint(low=0, high=4, size=(batch, M, N))
                                    .to(dtype=torch.float32)
                                    .cpu()
                                )
                                # Transpose it?
                                if transposing:
                                    a_cpu = a_cpu.transpose(-1, -2)
                                # Get convenient sizes
                                M, N = a_cpu.size()[-2:]
                                min_m_n = min(M, N)
                                max_m_n = max(M, N)
                                # Compute gold standard using double precision
                                U_double, s_double, Vh_double = torch.svd(
                                    a_cpu.to(torch.float64), full_matrices=full_matrix
                                )
                                # Make sure we got sensible results
                                assert torch.allclose(
                                    torch.matmul(U_double, torch.matmul(torch.diag_embed(s_double), Vh_double)),
                                    a_cpu,
                                    atol=atol,
                                    rtol=rtol,
                                )
                                # Prepare output tensors
                                U_cpu = torch.empty(
                                    (batch, M, N) if full_matrix else (batch, min_m_n, N),
                                    device=a_cpu.device,
                                    dtype=torch.float32,
                                )
                                Vh_cpu = torch.empty(
                                    (batch, M, N) if full_matrix else (batch, M, min_m_n),
                                    device=a_cpu.device,
                                    dtype=torch.float32,
                                )
                                S_cpu = torch.empty((batch, min_m_n), device=a_cpu.device, dtype=torch.float32)
                                # Move everything to the right device
                                a_gpu = a_cpu.to(dev_id)
                                U_gpu = torch.empty_like(U_cpu)
                                Vh_gpu = torch.empty_like(Vh_cpu)
                                S_gpu = torch.empty_like(S_cpu)
                                # Warm-up
                                _linalg_svd_compute[(1,)](
                                    U_gpu,
                                    S_gpu,
                                    Vh_gpu,
                                    a_gpu,
                                    M,
                                    N,
                                    min_m_n,
                                    max_m_n,
                                    a_gpu.stride(1),
                                    a_gpu.stride(2),
                                    U_gpu.stride(1),
                                    U_gpu.stride(2),
                                    Vh_gpu.stride(1),
                                    Vh_gpu.stride(2),
                                    BLOCK_SIZE_M=max(16, min(2047, M // 2)),
                                    BLOCK_SIZE_N=max(16, min(2047, N // 2)),
                                )
                                # Compute the normed error
                                torch.cuda.synchronize()
                                _linalg_svd_compute[(M, N)](
                                    U_gpu,
                                    S_gpu,
                                    Vh_gpu,
                                    a_gpu,
                                    M,
                                    N,
                                    min_m_n,
                                    max_m_n,
                                    a_gpu.stride(1),
