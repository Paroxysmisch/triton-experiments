import triton
import triton.language as tl
import torch

# Define the Triton kernel
@triton.jit
def _sgmv_expand_slice_kernel(
    A_ptr, B_ptr, C_ptr, lora_indices_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    M, N, K
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate block indices
    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N

    # Check boundary conditions
    mask_m = block_start_m + tl.arange(0, BLOCK_M) < M
    mask_n = block_start_n + tl.arange(0, BLOCK_N) < N

    # Initialize accumulation
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, K, BLOCK_K):
        # Load LoRA weights using lora_indices
        lora_idx = tl.load(lora_indices_ptr + k)
        A_block = tl.load(A_ptr + (block_start_m + tl.arange(0, BLOCK_M))[:, None] * K + lora_idx, mask=mask_m[:, None])
        B_block = tl.load(B_ptr + (lora_idx * N + block_start_n + tl.arange(0, BLOCK_N)), mask=mask_n)

        # Matrix multiplication
        acc += tl.dot(A_block, B_block)

    # Store results
    C_block_ptr = C_ptr + block_start_m * N + block_start_n
    tl.store(C_block_ptr, acc, mask=mask_m[:, None] & mask_n)

# Define the wrapper function
def _sgmv_expand_slice(A, B, lora_indices, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32):
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    lora_indices = lora_indices.contiguous()

    # Get dimensions
    M, K = A.shape
    _, N = B.shape

    # Validate dimensions
    assert K == lora_indices.shape[0], "Mismatch in dimensions of A and lora_indices"

    # Allocate output tensor
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Define grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch the kernel
    _sgmv_expand_slice_kernel[grid](
        A, B, C, lora_indices,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        M=M, N=N, K=K
    )

    return C

# Example usage
# A, B, lora_indices are assumed to be torch tensors with appropriate dimensions
# C = _sgmv_expand_slice(A, B, lora_indices)
