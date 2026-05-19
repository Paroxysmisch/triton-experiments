import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

@triton.jit
def matmul_kernel_persistent(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    sm_stride, ld_a, ld_b, ld_c,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    NUM_WARPS: tl.constexpr, NUM_SMS: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    ms_id = pid % NUM_SMS
    group_id = pid // NUM_SMS
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_size_m = min(num_pid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    first_pid_m = group_id * GROUP_size_m

    pid_m = first_pid_m + (group_size_m * (ms_id % (num_pid_in_group // GROUP_SIZE_M)))
    pid_n = (ms_id // (num_pid_m // GROUP_SIZE_M)) * BLOCK_SIZE_N

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))

    rem_k = K % BLOCK_SIZE_K
    pid_k_range = min(BLOCK_SIZE_K, rem_k)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, pid_k_range)

        a_ptrs = a_ptr + (offs_am[:, None] * ld_a + offs_k[None, :] * sm_stride)
        b_ptrs = b_ptr + (offs_k[:, None] * ld_b + offs_bn[None, :] * sm_stride)

        a = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)

        accumulator += tl.dot(a, b)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + ld_c * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    accumulator = accumulator.to(tl.float16 if ld_c == (N * 2) else tl.float32)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def matmul_persistent(a, b, output_dtype=None):
    assert a.is_contiguous()
    assert b.is_contiguous()

    a = a.to(torch.float16 if b.dtype == torch.bfloat16 else torch.float32)
    dtype = torch.bfloat16 if a.dtype == b.dtype == torch.bfloat16 else torch.float16

    if output_dtype is None:
        output_dtype = a.dtype

    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=output_dtype)
    grid_fn = lambda meta: (grid(meta["M"], meta["N"], meta["BLOCK_SIZE_M"], meta["BLOCK_SIZE_N"]) * meta["NUM_SMS"])

    with torch.cuda.device(a.device):
        matmul_kernel_persistent[grid_fn](
            a, b, c,
            a.shape[0], b.shape[1], a.shape[1],
            a.stride(0) if a.stride(0) >= a.stride(1) else a.stride(1),
            a.stride(0), b.stride(0), c.stride(0),
            BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, NUM_WARPS=4,
            NUM_SMS=torch.cuda.get_device_properties(a.device).multi_processor_count,
            GROUP_SIZE_M=4
        )
    return c
