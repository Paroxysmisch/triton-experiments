import triton
import triton.language as tl
import torch

@triton.jit
def load_reduce_kernel(
    x_ptr, 
    y_ptr,
    stride_xm, 
    stride_xn, 
    stride_y,
    M, 
    N,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Initialize partial maximum values with very low values
    partial_max = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    
    # Loop over columns in steps of BLOCK_N
    for col_start in range(0, N, BLOCK_N):
        col_idx = col_start + tl.arange(0, BLOCK_N)
        # Create a 2D index for the load
        # (BLOCK_M rows, BLOCK_N cols)
        # Broadcasting row_idx along axis=1, col_idx along axis=0
        load_mask = (row_idx < M) & (col_idx < N)
        x = tl.load(
            x_ptr + row_idx[:, None] * stride_xm + col_idx[None, :] * stride_xn,
            mask=load_mask,
            other=-float("inf"),
        )
        # Compute the max along columns for each row
        row_max = tl.max(x, axis=1)
        partial_max = tl.maximum(partial_max, row_max)
    
    # Store the final maximum value for each row
    store_mask = row_idx < M
    tl.store(y_ptr + row_idx * stride_y, partial_max, mask=store_mask)

def load_reduce(
    x: torch.Tensor, 
    BLOCK_M: int = 128, 
    BLOCK_N: int = 128
):
    M, N = x.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    grid = ( (M + BLOCK_M - 1) // BLOCK_M, )
    load_reduce_kernel[grid](
        x, 
        y, 
        x.stride(0), 
        x.stride(1),
        y.stride(0),
        M, 
        N,
        BLOCK_M=BLOCK_M, 
        BLOCK_N=BLOCK_N
    )
    # Verification
    torch.testing.assert_close(y, x.max(dim=1).values)
    return y

if __name__ == "__main__":
    # Example usage
    x = torch.randn(1024, 1024, device="cuda")
    y = load_reduce(x)
