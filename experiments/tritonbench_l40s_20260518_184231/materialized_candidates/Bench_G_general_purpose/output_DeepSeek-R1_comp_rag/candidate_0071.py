import triton
import triton.language as tl
import torch

@triton.jit
def load_reduce_kernel(
    x_ptr,
    y_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    M,  # Number of rows in the matrix
    N,  # Number of columns in the matrix
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    offs_m = row_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M  # Mask for rows within bounds

    # Initialize max values to -infinity
    row_max = tl.zeros(BLOCK_M, dtype=tl.float32) - float('inf')

    # Loop over columns in blocks of BLOCK_N
    for col_start in range(0, N, BLOCK_N):
        offs_n = col_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N  # Mask for columns within bounds
        mask = mask_m[:, None] & mask_n[None, :]  # 2D mask combining row and column masks

        # Create block pointer for the current block
        block_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(M, N),
            strides=(stride_xm, stride_xn),
            offsets=(row_start, col_start),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0)  # Row-major order
        )
        block = tl.load(block_ptr, mask=mask, other=-float('inf'))

        # Compute max along columns for the current block and update row_max
        current_max = tl.max(block, axis=1)
        row_max = tl.maximum(row_max, current_max)

    # Write the computed max values to the output
    tl.store(y_ptr + offs_m * stride_ym, row_max, mask=mask_m)

def load_reduce():
    # Test parameters
    M, N = 1024, 512  # Matrix dimensions
    BLOCK_M, BLOCK_N = 128, 32  # Block sizes (must be powers of two)

    # Generate random input matrix and initialize output vector
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    y = torch.empty(M, device='cuda', dtype=torch.float32)

    # Compute grid size (1D grid along rows)
    grid = (triton.cdiv(M, BLOCK_M),)

    # Launch the kernel
    load_reduce_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        stride_xm=x.stride(0),
        stride_xn=x.stride(1),
        stride_ym=y.stride(0),
        M=M,
        N=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    # Compute reference using PyTorch and check correctness
    y_ref = x.max(dim=1)[0]
    torch.testing.assert_close(y, y_ref, msg="Triton and Torch results differ")

# Run the test
if __name__ == "__main__":
    load_reduce()
    print("Test passed!")
