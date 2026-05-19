import torch
import triton
import triton.language as tl

@triton.jit
def spinning_lock_kernel(
    P,
    C,
    locks,
    num_sms,
    k: tl.constexpr,
    M,
    N,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Triton kernel to perform reduction using spinning lock mechanism
    pid = tl.program_id(axis=0)
    pid_m = pid // (N // BLOCK_SIZE_N)
    pid_n = pid % (N // BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(9):
        for j in range(tl.num_programs(1) // k):
            block_offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            block_offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            offset_m = block_offset_m + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_cm
            offset_n = block_offset_n + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_cn

            offset = offset_m[:, None] * M + offset_n[None, :]

            mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)

            if j == 0:
                p = tl.load(P + offset, mask=mask)
                acc += p
            else:
                tl.store(P + offset, acc, mask=mask)

                tl.atomic_xchg(locks + offset, 0)

        if pid % k == 0:
            break

        for _ in range(100):
            lock = tl.load(locks + P + offset)

            if tl.atomic_cas(locks + P + offset, lock, 1) == lock:
                break

        acc = tl.load(P + offset, mask=mask)

    tl.store(P + offset, acc, mask=mask)
    tl.atomic_xchg(locks + offset, 0)

    offset_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_cm = offset_cm < M
    mask_cn = offset_cn < N

    mask = mask_cm[:, None] & mask_cn[None, :]

    if pid_m == 0:
        p = tl.load(P + stride_cm * offset_cm[:, None] + stride_cn * offset_cn[None, :], mask=mask)
        tl.store(C + offset_cn[None, :], p, mask=mask_cn[None, :])

    if pid_n == 0:
        p = tl.load(P + stride_cm * offset_cm[None, :] + stride_cn * offset_cn[:, None], mask=mask)
        tl.store(C + offset_cm[:, None], p, mask=mask_cm[:, None])

def spinning_lock(P, C, num_sms, k):
    # Function to orchestrate the kernel launch
    M, N = P.shape

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)

    locks = torch.zeros_like(P, dtype=torch.int32, device="cuda")

    spinning_lock_kernel[grid](P, C, locks, num_sms, k, M, N, P.stride(0), P.stride(1), BLOCK_SIZE_M=128, BLOCK_SIZE_N=128)
