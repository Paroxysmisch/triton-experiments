import triton
import triton.language as tl

@triton.jit
def forward(q, k, v, q_scale, k_scale):
    Z, H, N_CTX, HEAD_DIM = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
    BLOCK_M = 32
    BLOCK_N = 32
    STAGE = 1

    # Determine the grid and block sizes
    num_blocks = Z * H * (N_CTX + BLOCK_M - 1) // BLOCK_M * (N_CTX + BLOCK_N - 1) // BLOCK_N
    grid_size = (num_blocks,)
    block_size = (BLOCK_M * BLOCK_N,)

    # Compute strides
    stride_qz, stride_qh, stride_qm, stride_qk = q.strides
    stride_kz, stride_kh, stride_kn, stride_kk = k.strides
    stride_vz, stride_vh, stride_vk, stride_vn = v.strides
    stride_oz, stride_oh, stride_om, stride_on = q.strides

    # Allocate output tensor
    out = tl.zeros_like(q)

    # Launch the Triton kernel
    _attn_fwd(q, k, v, q_scale, k_scale, out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, stride_oz, stride_oh, stride_om, stride_on, Z, H, N_CTX, HEAD_DIM, BLOCK_M, BLOCK_N, STAGE)

    return out
