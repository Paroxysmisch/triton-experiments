import triton
import triton.language as tl

# Triton kernel for forward pass of flash attention
@triton.jit
def _fwd_kernel(
    Q, K, V, O,  # Pointers to matrices
    stride_qb, stride_qh, stride_qm,  # Strides for Q
    stride_kb, stride_kh, stride_kn,  # Strides for K
    stride_vb, stride_vh, stride_vn,  # Strides for V
    stride_ob, stride_oh, stride_om,  # Strides for O
    N_CTX,  # Matrix width
    BLOCK_M: tl.constexpr,  # Block size for M
    BLOCK_N: tl.constexpr,  # Block size for N
    HEAD_DIM: tl.constexpr,  # Head dimension
    CAUSAL: tl.constexpr  # Causal masking
):
    # Matrix multiplication
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    hid = tl.program_id(axis=2)

    # Compute the block indices
    start_m = pid * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_b = bid
    offs_h = hid

    # Initialize pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qh),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kh, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vh),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=O,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_oh),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )

    # Load Q, K, V blocks
    q = tl.load(Q_block_ptr)
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)

    # Compute attention scores
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, HEAD_DIM, 16):
        qk = tl.dot(q[:, i:i+16], k[i:i+16, :])
        acc += qk

    # Apply scaling
    acc *= 1.0 / tl.sqrt(HEAD_DIM)

    # Apply causal masking if needed
    if CAUSAL:
        mask = (offs_m[:, None] >= offs_n[None, :]).to(tl.float32)
        acc = acc * mask

    # Softmax
    acc = tl.softmax(acc, axis=1)

    # Compute output
    o = tl.dot(acc, v)

    # Store the result
    tl.store(O_block_ptr, o)

# Wrapper function for flash attention
def flash_attn_triton(q, k, v, o, BLOCK_M, BLOCK_N, causal=False):
    # Get dimensions
    B, H, N_CTX, HEAD_DIM = q.shape

    # Launch the kernel
    grid = (N_CTX // BLOCK_M, B, H)
    _fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM, causal
    )

# Example usage
import torch

# Example tensors
B, H, N_CTX, HEAD_DIM = 2, 4, 128, 64
q = torch.randn((B, H, N_CTX, HEAD_DIM), device='cuda')
k = torch.randn((B, H, N_CTX, HEAD_DIM), device='cuda')
v = torch.randn((B, H, N_CTX, HEAD_DIM), device='cuda')
o = torch.empty((B, H, N_CTX, HEAD_DIM), device='cuda')

# Call the flash attention function
flash_attn_triton(q, k, v, o, BLOCK_M=16, BLOCK_N=16, causal=True)
