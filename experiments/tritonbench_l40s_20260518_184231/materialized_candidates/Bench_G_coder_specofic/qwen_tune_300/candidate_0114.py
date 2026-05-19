import torch
import triton
import triton.language as tl

@triton.jit
def spinning_lock_kernel(
    P,  # pointer to the input data
    C,  # pointer to the output data
    locks,  # pointer to the locks
    num_sms,  # number of_sms
    k,  # num_warps
    M,  # first dimension of the input
    N,  # second dimension of the input
    stride_cm,  # stride of the first dimension of the output
    stride_cn,  # stride of the second dimension of the output
    BLOCK_SIZE_M: tl.constexpr,  # block size for the first dimension
    BLOCK_SIZE_N: tl.constexpr,  # block size for the second dimension
):
    pid = tl.program_id(axis=0)
    pid_m = pid // num_sms
    pid_n = pid % num_sms
    accumulator = 0.0
    for i in range(0, 9):
        if pid % k == 0:
            for j in range(0, k):
                lock_id = (pid_n * k + j) % num_sms
                # attempt to acquire the lock
                if tl.atomic_cas(locks + lock_id, 0, 1) == 0:
                    acc = accumulator
                    accumulator += tl.load(
                        P + pid_m * stride_cm + ((pid_n * k + j) % num_sms) * stride_cn
                    )
                    # release the lock
                    tl.atomic_xchg(locks + lock_id, 0)
        else:
            lock_id = (pid_n * k + pid % k) % num_sms
            # release the lock
            tl.atomic_xchg(locks + lock_id, 0)
    if pid % k == 0:
        tl.store(C + pid_m * stride_cm + pid_n * stride_cn, accumulator)
    return

def spinning_lock(num_warps, device="cuda"):
    SIZE_M = 128
    SIZE_N = 128
    SIZE_K = 4

    # create a random tensor
    A = torch.randn((SIZE_M, SIZE_N), device=device, dtype=torch.float16)
    B = torch.randn((SIZE_N, SIZE_K), device=device, dtype=torch.float16)
    C = torch.empty((SIZE_M, SIZE_K), device=device, dtype=torch.float32)

    num_sms = 4 if device == "cuda" else 1
    grid = (triton.cdiv(SIZE_M, 1) * triton.cdiv(SIZE_N, 1),)

    locks = torch.zeros((triton.cdiv(SIZE_N, 1),), device=device, dtype=torch.int32)

    # enqueue kernel
    spinning_lock_kernel[grid](
        A,  # pointer to the first input
        C,  # pointer to the output
        locks,  # pointer to the locks
        num_sms,  # number of_sms
        SIZE_K,  # num_warps
        SIZE_M,  # first dimension of the input
        SIZE_N,  # second dimension of the input
        A.stride(0),  # stride of the first dimension of the first input
        A.stride(1),  # stride of the second dimension of the first input
        BLOCK_SIZE_M=1,  # block size for the first dimension
        BLOCK_SIZE_N=1,  # block size for the second dimension
        num_warps=num_warps,  # number of warps
    )
    # check result
    assert torch.allclose(C, torch.matmul(A, B))
    return
