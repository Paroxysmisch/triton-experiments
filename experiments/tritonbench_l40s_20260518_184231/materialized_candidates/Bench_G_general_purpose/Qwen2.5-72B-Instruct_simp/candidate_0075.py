import triton
import triton.language as tl

@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr, o_ptr,
    BT, BK, BV, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr
):
    # Compute the block indices
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Compute the block bounds
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Compute the block pointers
    q_ptrs = q_ptr + (offs_m[:, None] * H * BLOCK_DMODEL + offs_h[None, :] * BLOCK_DMODEL + offs_d[None, :]) * BT
    k_ptrs = k_ptr + (offs_n[:, None] * H * BLOCK_DMODEL + offs_h[None, :] * BLOCK_DMODEL + offs_d[None, :]) * BK
    v_ptrs = v_ptr + (offs_n[:, None] * H * BLOCK_DMODEL + offs_h[None, :] * BLOCK_DMODEL + offs_d[None, :]) * BV
    h_ptrs = h_ptr + (offs_m[:, None] * H * BLOCK_DMODEL + offs_h[None, :] * BLOCK_DMODEL + offs_d[None, :]) * BT
    o_ptrs = o_ptr + (offs_m[:, None] * H * BLOCK_DMODEL + offs_h[None, :] * BLOCK_DMODEL + offs_d[None, :]) * BT

    # Load the data
    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)
    h = tl.load(h_ptrs)

    # Compute the dot product
    logits = tl.dot(q, k, trans_b=True)  # (BLOCK_M, BLOCK_N)
    logits = logits * (1.0 / tl.sqrt(tl.float32(BLOCK_DMODEL)))

    # Apply exponential transformation
    logits = tl.exp(logits)

    # Compute the weighted sum
    v = tl.broadcast_to(v, (BLOCK_M, BLOCK_N, BLOCK_DMODEL))
    o = tl.sum(logits[:, :, None] * v, axis=1)  # (BLOCK_M, BLOCK_DMODEL)

    # Apply auxiliary tensor
    o = o + h

    # Store the result
    tl.store(o_ptrs, o)

import torch

def chunk_fwd_o_fn(q, k, v, h, o, BT, BK, BV, H, N_CTX, BLOCK_M, BLOCK_DMODEL, BLOCK_N, BLOCK_H):
    # Get the grid dimensions
    grid = (N_CTX // BLOCK_M, H // BLOCK_H, N_CTX // BLOCK_N)

    # Launch the kernel
    chunk_simple_gla_fwd_kernel_o[grid](
        q, k, v, h, o,
        BT, BK, BV, H, N_CTX,
        BLOCK_M, BLOCK_DMODEL, BLOCK_N, BLOCK_H
    )

# Example usage
if __name__ == "__main__":
    # Example dimensions
    BT = 1024
    BK = 1024
    BV = 1024
    H = 16
    N_CTX = 512
    BLOCK_M = 32
    BLOCK_DMODEL = 64
    BLOCK_N = 32
    BLOCK_H = 16

    # Create example tensors
    q = torch.randn((N_CTX, H, BLOCK_DMODEL), device='cuda')
    k = torch.randn((N_CTX, H, BLOCK_DMODEL), device='cuda')
    v = torch.randn((N_CTX, H, BLOCK_DMODEL), device='cuda')
    h = torch.randn((N_CTX, H, BLOCK_DMODEL), device='cuda')
    o = torch.zeros((N_CTX, H, BLOCK_DMODEL), device='cuda')

    # Call the wrapper function
    chunk_fwd_o_fn(q, k, v, h, o, BT, BK, BV, H, N_CTX, BLOCK_M, BLOCK_DMODEL, BLOCK_N, BLOCK_H)

    # Print the result
    print(o)
