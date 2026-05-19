import triton
import triton.language as tl
import torch

@triton.jit
def kernel(
    M, Out,
    matrix_stridex, matrix_stridey,
    out_stridex, out_stridey,
    SIZE_M, D_HEAD,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_block_x = (SIZE_M + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_block_y = (D_HEAD + BLOCK_SIZE - 1) // BLOCK_SIZE

    block_x = pid % num_block_x
    block_y = pid // num_block_x

    # Offsets for rows/cols in the original and transposed matrix
    offs_x = block_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_y = block_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Create a 2D mask to guard accesses beyond matrix bounds
    mask = (offs_x[:, None] < SIZE_M) & (offs_y[None, :] < D_HEAD)

    # Pointer arithmetic for M
    m_ptrs = M + (offs_x[:, None] * matrix_stridex) + (offs_y[None, :] * matrix_stridey)
    # Load
    vals = tl.load(m_ptrs, mask=mask, other=0.0)

    # Pointer arithmetic for Out (transposed indices)
    out_ptrs = Out + (offs_y[None, :] * out_stridex) + (offs_x[:, None] * out_stridey)
    # Store
    tl.store(out_ptrs, vals, mask=mask)

def wrapper(SIZE_M, D_HEAD, BLOCK_SIZE=32):
    M = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device='cuda')
    Out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device='cuda')

    num_block_x = (SIZE_M + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_block_y = (D_HEAD + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_block_x * num_block_y,)

    kernel[grid](
        M, Out,
        M.stride(0), M.stride(1),
        Out.stride(0), Out.stride(1),
        SIZE_M, D_HEAD,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return Out
