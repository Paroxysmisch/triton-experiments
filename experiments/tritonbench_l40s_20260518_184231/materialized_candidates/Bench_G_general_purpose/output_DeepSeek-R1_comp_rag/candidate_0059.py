import triton
import triton.language as tl

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    # Tensors
    q_ptr, k_ptr, g_ptr, A_ptr,
    # Scalars
    scale,
    chunk_size,
    batch_size, n_heads, d_model,
    # Strides for Q
    q_batch_stride, q_head_stride, q_chunk_stride, q_block_stride,
    # Strides for K
    k_batch_stride, k_head_stride, k_chunk_stride, k_block_stride,
    # Strides for G
    g_batch_stride, g_head_stride, g_chunk_stride, g_block_stride,
    # Strides for A
    A_batch_stride, A_head_stride, A_chunk_i_stride, A_chunk_j_stride, A_block_stride,
    # Tile parameters
    BLOCK_K: tl.constexpr, SUB_BLOCK: tl.constexpr,
    # Meta-parameters
    IS_UPPER_TRI: tl.constexpr, ALLOW_TF32: tl.constexpr
):
    # 3D launch grid: [batch, heads, chunks, chunks]
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk_i = tl.program_id(2)
    pid_chunk_j = tl.program_id(3)
    
    # Skip lower triangle if upper triangular mode
    if IS_UPPER_TRI and (pid_chunk_i <= pid_chunk_j):
        return

    # Offset calculations for Q, K, G blocks
    q_off = pid_batch * q_batch_stride + pid_head * q_head_stride + pid_chunk_i * q_chunk_stride
    k_off = pid_batch * k_batch_stride + pid_head * k_head_stride + pid_chunk_j * k_chunk_stride
    g_off = pid_batch * g_batch_stride + pid_head * g_head_stride + pid_chunk_i * g_chunk_stride

    # Initialize accumulator
    acc = tl.zeros((SUB_BLOCK, SUB_BLOCK), dtype=tl.float32)
    
    # Load blocks and compute
    for k in range(0, d_model, BLOCK_K):
        k_mask = k + tl.arange(0, BLOCK_K) < d_model
        # Load Q block [SUB_BLOCK, BLOCK_K]
        q = tl.load(q_ptr + q_off + k * q_block_stride + tl.arange(0, SUB_BLOCK)[:, None] * chunk_size,
                    mask=k_mask[None, :] & (tl.arange(0, SUB_BLOCK)[:, None] < chunk_size), other=0.0)
        # Load K block [BLOCK_K, SUB_BLOCK]
        k = tl.load(k_ptr + k_off + k * k_block_stride + tl.arange(0, SUB_BLOCK)[None, :],
                    mask=k_mask[:, None] & (tl.arange(0, SUB_BLOCK)[None, :] < chunk_size), other=0.0)
        # Load G block [SUB_BLOCK, BLOCK_K]
        g = tl.load(g_ptr + g_off + k * g_block_stride + tl.arange(0, SUB_BLOCK)[:, None] * chunk_size,
                    mask=k_mask[None, :] & (tl.arange(0, SUB_BLOCK)[:, None] < chunk_size), other=1.0)  # 1.0 if no gate
        
        # Compute scaled and gated values
        q_scaled = tl.exp(q * scale) * g
        k_scaled = tl.exp(k * scale)
        acc += tl.dot(q_scaled, k_scaled, allow_tf32=ALLOW_TF32)
    
    # Store result to A
    a_off = (pid_batch * A_batch_stride + pid_head * A_head_stride + 
             pid_chunk_i * A_chunk_i_stride + pid_chunk_j * A_chunk_j_stride)
    tl.store(A_ptr + a_off + tl.arange(0, SUB_BLOCK)[:, None] * A_block_stride + tl.arange(0, SUB_BLOCK)[None, :],
             acc, mask=(tl.arange(0, SUB_BLOCK)[:, None] < chunk_size) & (tl.arange(0, SUB_BLOCK)[None, :] < chunk_size))

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra(
    q_ptr, g_ptr, A_ptr, scale,
    chunk_size, batch_size, n_heads, d_model,
    # Strides similar to inter kernel...
    BLOCK_K: tl.constexpr, SUB_BLOCK: tl.constexpr,
    ALLOW_TF32: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    
    q_off = pid_batch * q_batch_stride + pid_head * q_head_stride + pid_chunk * q_chunk_stride
    g_off = pid_batch * g_batch_stride + pid_head * g_head_stride + pid_chunk * g_chunk_stride

    acc = tl.zeros((SUB_BLOCK, SUB_BLOCK), dtype=tl.float32)
    
    for k in range(0, d_model, BLOCK_K):
        q = tl.load(q_ptr + q_off + k * q_block_stride + tl.arange(0, SUB_BLOCK)[:, None] * chunk_size,
                    mask=(k + tl.arange(0, BLOCK_K)[None, :] < d_model) & (tl.arange(0, SUB_BLOCK)[:, None] < chunk_size))
        g = tl.load(g_ptr + g_off + k * g_block_stride + tl.arange(0, SUB_BLOCK)[:, None] * chunk_size,
                    mask=(k + tl.arange(0, BLOCK_K)[None, :] < d_model) & (tl.arange(0, SUB_BLOCK)[:, None] < chunk_size))
        
        q_scaled = tl.exp(q * scale) * g
        # Intra-chunk uses outer product
        acc += tl.dot(q_scaled, tl.trans(q_scaled), allow_tf32=ALLOW_TF32)
    
    a_off = (pid_batch * A_batch_stride + pid_head * A_head_stride + 
             pid_chunk * A_chunk_i_stride + pid_chunk * A_chunk_j_stride)
    tl.store(A_ptr + a_off + tl.arange(0, SUB_BLOCK)[:, None] * A_block_stride + tl.arange(0, SUB_BLOCK)[None, :], acc)

@triton.jit
def chunk_gla_fwd_kernel_o(
    A_ptr, v_ptr, o_ptr,
    scale, chunk_size, d_model,
    # Strides...
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ALLOW_TF32: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    
    a_off = pid_batch * A_batch_stride + pid_head * A_head_stride + pid_chunk * A_chunk_i_stride
    v_off = pid_batch * v_batch_stride + pid_head * v_head_stride + pid_chunk * v_chunk_stride
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for c in range(0, chunk_size):
        a = tl.load(A_ptr + a_off + c * A_block_stride + tl.arange(0, BLOCK_M)[:, None])
        v = tl.load(v_ptr + v_off + c * v_block_stride + tl.arange(0, BLOCK_N)[None, :])
        acc += tl.dot(a, v, allow_tf32=ALLOW_TF32)
    
    o_off = pid_batch * o_batch_stride + pid_head * o_head_stride + pid_chunk * o_chunk_stride
    tl.store(o_ptr + o_off + tl.arange(0, BLOCK_M)[:, None] * o_block_stride + tl.arange(0, BLOCK_N)[None, :], acc)

def chunk_gla_fwd_chunk_attention(
    q: torch.Tensor, k: torch.Tensor, g: torch.Tensor,
    chunk_size: int, scale: float = 1.0
) -> torch.Tensor:
    batch, n_heads, seq_len, d_model = q.shape
    num_chunks = seq_len // chunk_size
    A = torch.zeros((batch, n_heads, num_chunks, num_chunks, chunk_size, chunk_size), device=q.device)
    
    # Configure kernel launches
    def grid(meta):
        return (batch, n_heads, num_chunks, num_chunks)
    
    # Launch inter-chunk kernel
    chunk_gla_fwd_A_kernel_intra_sub_inter[grid](
        q, k, g, A, scale, chunk_size, batch, n_heads, d_model,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        A.stride(0), A.stride(1), A.stride(2), A.stride(3), A.stride(4),
        BLOCK_K=64, SUB_BLOCK=32, IS_UPPER_TRI=True, ALLOW_TF32=True
    )
    
    # Launch intra-chunk kernel
    def intra_grid(meta):
        return (batch, n_heads, num_chunks)
    chunk_gla_fwd_A_kernel_intra_sub_intra[intra_grid](
        q, g, A, scale, chunk_size, batch, n_heads, d_model,
        # Strides...
        BLOCK_K=64, SUB_BLOCK=32, ALLOW_TF32=True
    )
    
    return A

def chunk_gla_fwd_output(
    A: torch.Tensor, v: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    batch, n_heads, num_chunks, _, _, _ = A.shape
    o = torch.empty_like(v)
    
    def grid(meta):
        return (batch, n_heads, num_chunks)
    
    chunk_gla_fwd_kernel_o[grid](
        A, v, o, chunk_size, v.size(-1),
        A.stride(0), A.stride(1), A.stride(2), A.stride(3), A.stride(4),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        BLOCK_M=32, BLOCK_N=64, ALLOW_TF32=True
    )
    return o
