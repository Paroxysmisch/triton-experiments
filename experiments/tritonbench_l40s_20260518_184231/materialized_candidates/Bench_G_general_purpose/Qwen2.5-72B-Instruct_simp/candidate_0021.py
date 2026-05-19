import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, O_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Compute the program ID in a 3D grid
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_id = pid // num_pid_in_batch
    pid = pid % num_pid_in_batch
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Compute the offsets for the current tile
    offs_am = (batch_id * stride_am + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None]
    offs_bn = (batch_id * stride_bn + pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the current tile
    A_tile_ptr = A_ptr + (offs_am * stride_ak + offs_k[None, :])
    B_tile_ptr = B_ptr + (offs_k[:, None] * stride_bn + offs_bn)

    # Accumulate the result in a shared memory tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load the current tile of A and B
        a = tl.load(A_tile_ptr)
        b = tl.load(B_tile_ptr)

        # Perform the matrix multiplication
        acc += tl.dot(a, b)

        # Move to the next tile
        A_tile_ptr += BLOCK_K * stride_ak
        B_tile_ptr += BLOCK_K * stride_bk

    # Store the result in the output tensor
    offs_om = (batch_id * stride_om + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None]
    offs_on = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    O_tile_ptr = O_ptr + (offs_om * stride_on + offs_on)
    tl.store(O_tile_ptr, acc)

import torch
import triton
import triton.language as tl

def bmm(A: torch.Tensor, B: torch.Tensor, O: torch.Tensor):
    # Ensure the input tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    O = O.contiguous()

    # Extract the dimensions
    batch, M, K = A.shape
    batch, K, N = B.shape

    # Define the grid and block sizes
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 16
    grid = (batch * tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N),)

    # Launch the kernel
    bmm_kernel[grid](
        A, B, O,
        M, N, K,
        A.stride(1), A.stride(2), B.stride(1), B.stride(2),
        O.stride(1), O.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_K
    )

# Example usage
batch, M, K, N = 2, 128, 128, 128
A = torch.randn((batch, M, K), device='cuda')
B = torch.randn((batch, K, N), device='cuda')
O = torch.empty((batch, M, N), device='cuda')

bmm(A, B, O)

# Verify the result
expected = torch.bmm(A, B)
torch.testing.assert_allclose(O, expected)
