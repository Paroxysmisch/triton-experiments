import triton
import triton.language as tl

@triton.jit
def rms_matmul_rbe(x_ptr, w_ptr, c_ptr, rms_w_ptr, rotary_ptr, 
                   M, N, K, stride_xm, stride_xk, stride_wk, stride_wn, 
                   stride_cm, stride_cn, apply_rotary, BLOCK_SIZE: tl.constexpr):
    # Define the block indices
    pid = tl.program_id(axis=0)
    m_offset = pid * BLOCK_SIZE

    # Offsets for x, w, and c
    x_offset = m_offset * stride_xm
    c_offset = m_offset * stride_cm

    # Load the block of x
    x = tl.load(x_ptr + x_offset + tl.arange(0, BLOCK_SIZE)[:, None] * stride_xk, mask=tl.arange(0, BLOCK_SIZE)[:, None] < M - m_offset, other=0.0)

    # Compute RMS normalization for x
    rms_x = tl.sqrt(tl.sum(x * x, axis=1) / K)

    # Load the block of w
    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_wn, mask=tl.arange(0, BLOCK_SIZE)[:, None] < N, other=0.0)

    # Compute RMS normalization for w
    rms_w = tl.load(rms_w_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N, other=1.0)

    # Optionally apply rotary embeddings
    if apply_rotary:
        rotary = tl.load(rotary_ptr + tl.arange(0, BLOCK_SIZE)[:, None], mask=tl.arange(0, BLOCK_SIZE)[:, None] < N, other=0.0)
        x = x * rotary

    # Perform matrix multiplication with RMS normalization
    x_normalized = x / rms_x[:, None]
    w_normalized = w * rms_w[None, :]
    c = tl.dot(x_normalized, w_normalized)

    # Store the result
    tl.store(c_ptr + c_offset + tl.arange(0, BLOCK_SIZE)[:, None] * stride_cn, c, mask=tl.arange(0, BLOCK_SIZE)[:, None] < M - m_offset)


def rms_matmul_rbe_qkv_wrapper(Q, K, V, W_q, W_k, W_v, rms_w_q, rms_w_k, rms_w_v, rotary_embeddings=None, apply_rotary=False):
    # Extract dimensions
    M, K = Q.shape
    _, N = W_q.shape

    # Define strides
    stride_qm, stride_qk = Q.stride()
    stride_wqk, stride_wqn = W_q.stride()
    stride_cm, stride_cn = M, N

    # Allocate output
    C_q = torch.empty((M, N), device=Q.device, dtype=Q.dtype)
    C_k = torch.empty((M, N), device=K.device, dtype=K.dtype)
    C_v = torch.empty((M, N), device=V.device, dtype=V.dtype)

    # Launch kernel for Q
    triton.launch(kernel=rms_matmul_rbe, grid=(M // BLOCK_SIZE,),
                  args=[Q, W_q, C_q, rms_w_q, rotary_embeddings, M, N, K, stride_qm, stride_qk, stride_wqk, stride_wqn, stride_cm, stride_cn, apply_rotary],
                  num_warps=4, BLOCK_SIZE=BLOCK_SIZE)

    # Launch kernel for K
    triton.launch(kernel=rms_matmul_rbe, grid=(M // BLOCK_SIZE,),
                  args=[K, W_k, C_k, rms_w_k, rotary_embeddings, M, N, K, stride_qm, stride_qk, stride_wqk, stride_wqn, stride_cm, stride_cn, apply_rotary],
                  num_warps=4, BLOCK_SIZE=BLOCK_SIZE)

    # Launch kernel for V
    triton.launch(kernel=rms_matmul_rbe, grid=(M // BLOCK_SIZE,),
                  args=[V, W_v, C_v, rms_w_v, rotary_embeddings, M, N, K, stride_qm, stride_qk, stride_wqk, stride_wqn, stride_cm, stride_cn, apply_rotary],
                  num_warps=4, BLOCK_SIZE=BLOCK_SIZE)

    return C_q, C_k, C_v
