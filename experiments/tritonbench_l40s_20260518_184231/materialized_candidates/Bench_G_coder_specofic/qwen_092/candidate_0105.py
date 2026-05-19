import triton
import triton.language as tl
import torch

def rbe_triton_wrapper(x: torch.Tensor, theta: float = 10000.0):
    # Get input tensor shape
    batch, M, K = x.shape

    # Define block sizes
    BLOCK_SIZE_M = 2
    BLOCK_SIZE_K = 1024

    # Create output tensor
    out = torch.zeros_like(x)

    # Define the Triton kernel
    @triton.jit
    def rbe_triton_kernel(
        x_ptr: tl.tensor,
        out_ptr: tl.tensor,
        M: tl.int32,
        K: tl.int32,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        theta: tl.constexpr
    ):
        # Kernel code here (copied from above)

    # Launch the Triton kernel
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(K, BLOCK_SIZE_K)
    grid_size = grid_m * grid_n
    rbe_triton_kernel[grid_size, BLOCK_SIZE_M * BLOCK_SIZE_K](
        x_ptr=x.contiguous().data_ptr(),
        out_ptr=out.contiguous().data_ptr(),
        M=M,
        K=K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        theta=theta
    )

    return out
