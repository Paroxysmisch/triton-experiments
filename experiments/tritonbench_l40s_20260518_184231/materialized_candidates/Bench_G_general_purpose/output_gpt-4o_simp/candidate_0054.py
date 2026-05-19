import triton
import triton.language as tl

@triton.jit
def chunk_gated_abc_fwd_kernel_cum(s_ptr, o_ptr, stride_s, stride_o, T, S, BT, BS, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    
    # Calculate block start positions
    row_start = pid * BLOCK_SIZE
    col_start = bid * BLOCK_SIZE

    # Load source data
    s = tl.load(s_ptr + row_start * stride_s + col_start, mask=(row_start < T) & (col_start < S))
    
    # Initialize output with zeros
    o = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)

    # Upper triangular mask
    m_s = tl.arange(0, BLOCK_SIZE)[:, None] <= tl.arange(0, BLOCK_SIZE)

    # Cumulative computation
    for i in range(BLOCK_SIZE):
        if m_s[i, i]:
            o[i, :] = tl.sum(s[i, :], axis=0)
    
    # Store the result
    tl.store(o_ptr + row_start * stride_o + col_start, o, mask=(row_start < T) & (col_start < S))

@triton.jit
def chunk_gated_abc_fwd_kernel_h(k_ptr, v_ptr, g_ptr, h_ptr, stride_k, stride_v, stride_g, stride_h, T, S, BT, BS, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    
    # Calculate block start positions
    row_start = pid * BLOCK_SIZE
    col_start = bid * BLOCK_SIZE

    # Load key, value, and gate data
    k = tl.load(k_ptr + row_start * stride_k + col_start, mask=(row_start < T) & (col_start < S))
    v = tl.load(v_ptr + row_start * stride_v + col_start, mask=(row_start < T) & (col_start < S))
    g = tl.load(g_ptr + row_start * stride_g + col_start, mask=(row_start < T) & (col_start < S))

    # Initialize output
    h = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)

    # Gated accumulation
    for i in range(BLOCK_SIZE):
        h[i, :] += g[i, :] * (k[i, :] @ v[i, :])

    # Store the result
    tl.store(h_ptr + row_start * stride_h + col_start, h, mask=(row_start < T) & (col_start < S))

def fwd_pre(s, T, S, BT, BS):
    # Set up grid for chunk_gated_abc_fwd_kernel_cum
    grid = (triton.cdiv(T, BS), triton.cdiv(S, BS))
    o = torch.empty_like(s)
    
    # Launch the kernel
    chunk_gated_abc_fwd_kernel_cum[grid](s, o, s.stride(0), o.stride(0), T, S, BT, BS, BLOCK_SIZE=BS)
    
    return o

def fwd_inner(k, v, g, h, T, S, BT, BS):
    # Set up grid for chunk_gated_abc_fwd_kernel_h
    grid = (triton.cdiv(T, BS), triton.cdiv(S, BS))
    h_out = torch.empty_like(h)
    
    # Launch the kernel
    chunk_gated_abc_fwd_kernel_h[grid](k, v, g, h_out, k.stride(0), v.stride(0), g.stride(0), h_out.stride(0), T, S, BT, BS, BLOCK_SIZE=BS)
    
    return h_out
