import triton
import triton.language as tl

@triton.jit
def symmetric_mm_and_abs_sum_kernel(
    A_ptr, C_ptr, alpha, beta, M, N, stride_am, stride_an, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = (pid % num_pid_in_group) + group_id * num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None]
    offs_an = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_N))[None, :]
    offs_cm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None]
    offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :]

    A = tl.load(A_ptr + offs_am * stride_am + offs_an * stride_an)
    A_T = tl.load(A_ptr + offs_an * stride_am + offs_am * stride_an)
    C = tl.load(C_ptr + offs_cm * stride_cm + offs_cn * stride_cn)

    # Compute the symmetric product
    result = alpha * tl.dot(A, A_T) + beta * C

    # Store the result back to C
    tl.store(C_ptr + offs_cm * stride_cm + offs_cn * stride_cn, result)

    # Compute the sum of absolute values
    abs_sum = tl.sum(tl.abs(result), axis=None)
    tl.atomic_add(abs_sum_ptr + 0, abs_sum)

import torch
import triton
import triton.language as tl

def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Ensure the input tensors are on the same device
    device = A.device
    assert A.device == C.device, "A and C must be on the same device"

    # Get the dimensions of the input tensors
    M, N = A.shape

    # Allocate a tensor to store the sum of absolute values
    abs_sum = torch.zeros(1, device=device, dtype=torch.float32)

    # Define the grid and block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    # Launch the Triton kernel
    symmetric_mm_and_abs_sum_kernel[grid](
        A, C, alpha, beta, M, N, A.stride(0), A.stride(1), C.stride(0), C.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, abs_sum
    )

    return abs_sum

# Example usage
A = torch.randn(128, 128, device='cuda')
C = torch.randn(128, 128, device='cuda')
alpha = 1.0
beta = 0.5
result = symmetric_mm_and_abs_sum(A, C, alpha, beta)
print(result)
