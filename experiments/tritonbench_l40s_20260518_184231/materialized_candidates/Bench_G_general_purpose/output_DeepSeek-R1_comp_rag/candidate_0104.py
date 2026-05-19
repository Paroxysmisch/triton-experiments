import triton
import triton.language as tl
import torch

@triton.jit
def kernel(
    M_ptr, Out_ptr,
    matrix_stridex, matrix_stridey,
    out_stridex, out_stridey,
    SIZE_M, D_HEAD,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    num_blocks_n = tl.cdiv(D_HEAD, BLOCK_SIZE_N)
    num_blocks_m = tl.cdiv(SIZE_M, BLOCK_SIZE_M)
    
    i_block = pid // num_blocks_n
    j_block = pid % num_blocks_n

    offs_im = i_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_jn = j_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_im = offs_im < SIZE_M
    mask_jn = offs_jn < D_HEAD

    m_ptrs = M_ptr + offs_im[:, None] * matrix_stridex + offs_jn[None, :] * matrix_stridey
    block = tl.load(m_ptrs, mask=mask_im[:, None] & mask_jn[None, :], other=0.0)

    block_t = tl.trans(block)

    transposed_offs_i = offs_jn
    transposed_offs_j = offs_im

    out_ptrs = Out_ptr + transposed_offs_i[:, None] * out_stridex + transposed_offs_j[None, :] * out_stridey

    mask_transposed_i = transposed_offs_i < D_HEAD
    mask_transposed_j = transposed_offs_j < SIZE_M
    store_mask = mask_transposed_i[:, None] & mask_transposed_j[None, :]

    tl.store(out_ptrs, block_t, mask=store_mask)

def wrapper():
    SIZE_M = 1024  # Example dimension M
    D_HEAD = 512   # Example dimension D_HEAD (N)
    matrix = torch.randn((SIZE_M, D_HEAD), device='cuda', dtype=torch.float16)
    out = torch.zeros((D_HEAD, SIZE_M), device='cuda', dtype=torch.float16)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64

    grid_m = (SIZE_M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (D_HEAD + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m * grid_n, )

    kernel[grid](
        matrix, out,
        matrix.stride(0), matrix.stride(1),
        out.stride(0), out.stride(1),
        SIZE_M, D_HEAD,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    return out
