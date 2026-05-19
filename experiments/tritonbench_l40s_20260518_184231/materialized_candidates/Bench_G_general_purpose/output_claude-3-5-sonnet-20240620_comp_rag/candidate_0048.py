import torch
import triton
import triton.language as tl

@triton.jit
def chunk_gated_abc_fwd_kernel_cum(
    s, o,                                    # pointers to input and output tensors
    s_s_h, s_s_t, s_s_d,                    # strides for the tensor dimensions
    T: tl.constexpr, S: tl.constexpr,       # total sizes
    BT: tl.constexpr, BS: tl.constexpr      # block sizes
):
    # Get program IDs for the current block
    i_s = tl.program_id(0)  # S dimension
    i_t = tl.program_id(1)  # T dimension
    i_bh = tl.program_id(2) # Batch * Head dimension
    
    # Create offset indices for the block
    o_i = tl.arange(0, BT)
    
    # Create mask for cumulative operation (lower triangular)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
    m_s = m_s.to(tl.float32)
    
    # Create block pointers for efficient memory access
    p_s = tl.make_block_ptr(
        base=s + i_bh * s_s_h,
        shape=(T, S),
        strides=(s_s_t, s_s_d),
        offsets=(i_t * BT, i_s * BS),
        block_shape=(BT, BS),
        order=(1, 0)
    )
    
    p_o = tl.make_block_ptr(
        base=o + i_bh * s_s_h,
        shape=(T, S),
        strides=(s_s_t, s_s_d),
        offsets=(i_t * BT, i_s * BS),
        block_shape=(BT, BS),
        order=(1, 0)
    )
    
    # Load input block and convert to float32
    b_s = tl.load(p_s, boundary_check=(0, 1))
    b_s = b_s.to(tl.float32)
    
    # Compute cumulative sum using matrix multiplication
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    
    # Store result
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

def fwd_pre(g, B, H, T, S, BT):
    """
    Wrapper function to prepare and launch the cumulative kernel
    
    Args:
        g: Input tensor
        B: Batch size
        H: Number of heads
        T: Sequence length
        S: Feature dimension
        BT: Block size for T dimension
    """
    # Calculate number of blocks needed
    NT = triton.cdiv(T, BT)
    
    # Create output tensor
    g_org, g = g, torch.empty_like(g, dtype=torch.float)
    
    # Define grid for kernel launch
    BS = 32  # Define block size for S dimension
    grid = lambda meta: (
        triton.cdiv(meta['S'], meta['BS']),
        NT,
        B * H
    )
    
    # Launch kernel
    chunk_gated_abc_fwd_kernel_cum[grid](
        g_org, g,
        g.stride(1), g.stride(2), g.stride(3),
        T=T, S=S,
        BT=BT, BS=BS
    )
    
    return g
