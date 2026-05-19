import triton
import triton.language as tl
import torch

# Triton kernel for matrix multiplication
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    # Program IDs for grid of execution
    pid_m = tl.program_id(0)  # Row ID
    pid_n = tl.program_id(1)  # Column ID

    # Compute the start indices for the current block
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Initialize accumulator for the block of C
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_SIZE_K chunks
    for k in range(0, K, BLOCK_SIZE_K):
        # Load a block of A and B from global memory
        A_block = tl.load(
            A_ptr + (block_start_m + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_am +
            (k + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_ak,
            mask=(block_start_m + tl.arange(0, BLOCK_SIZE_M))[:, None] < M,
            other=0.0
        )
        B_block = tl.load(
            B_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * stride_bk +
            (block_start_n + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_bn,
            mask=(block_start_n + tl.arange(0, BLOCK_SIZE_N))[None, :] < N,
            other=0.0
        )

        # Perform the dot product for the current block
        accumulator += tl.dot(A_block, B_block)

    # Apply optional activation function (leaky ReLU)
    if ACTIVATION == "leaky_relu":
        accumulator = tl.where(accumulator > 0, accumulator, 0.01 * accumulator)

    # Write the result to the output matrix C
    C_block_ptr = C_ptr + (block_start_m + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_cm + \
                  (block_start_n + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_cn
    mask = (block_start_m + tl.arange(0, BLOCK_SIZE_M))[:, None] < M & \
           (block_start_n + tl.arange(0, BLOCK_SIZE_N))[None, :] < N
    tl.store(C_block_ptr, accumulator, mask=mask)


# Python wrapper for the Triton kernel
def matmul(A, B, activation=None):
    """
    Perform matrix multiplication C = A x B with optional activation.
    
    Args:
        A (torch.Tensor): Input matrix A of shape (M, K).
        B (torch.Tensor): Input matrix B of shape (K, N).
        activation (str, optional): Activation function to apply. Supported: "leaky_relu".
    
    Returns:
        torch.Tensor: Resulting matrix C of shape (M, N).
    """
    # Validate input shapes
    assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D tensors."
    assert A.shape[1] == B.shape[0], "Inner dimensions of A and B must match."

    # Get shapes
    M, K = A.shape
    K, N = B.shape

    # Allocate output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    # Define grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Set activation
    ACTIVATION = activation if activation in ["leaky_relu"] else None

    # Launch the Triton kernel
    matmul_kernel[grid](
        A_ptr=A.data_ptr(),
        B_ptr=B.data_ptr(),
        C_ptr=C.data_ptr(),
        M=M, N=N, K=K,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_bk=B.stride(0), stride_bn=B.stride(1),
        stride_cm=C.stride(0), stride_cn=C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        ACTIVATION=ACTIVATION
    )

    return C
