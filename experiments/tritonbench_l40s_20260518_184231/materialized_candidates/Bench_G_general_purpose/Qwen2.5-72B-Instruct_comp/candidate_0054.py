import triton
import triton.language as tl

@triton.jit
def chunk_gated_abc_fwd_kernel_cum(
    s_ptr, o_ptr, T, S, BT, BS, stride_s_T, stride_s_S, stride_o_T, stride_o_S,
    BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_S: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_T
    offsets_t = block_start + tl.arange(0, BLOCK_SIZE_T)
    offsets_s = tl.arange(0, BLOCK_SIZE_S)
    mask = (offsets_t < T)[:, None] & (offsets_s < S)[None, :]

    s_block_ptr = tl.make_block_ptr(
        base=s_ptr, shape=(T, S), strides=(stride_s_T, stride_s_S),
        offsets=(block_start, 0), block_shape=(BLOCK_SIZE_T, BLOCK_SIZE_S), order=(1, 0)
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr, shape=(T, S), strides=(stride_o_T, stride_o_S),
        offsets=(block_start, 0), block_shape=(BLOCK_SIZE_T, BLOCK_SIZE_S), order=(1, 0)
    )

    s = tl.load(s_block_ptr, mask=mask, other=0.0)
    o = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_S), dtype=tl.float32)

    for i in range(BLOCK_SIZE_S):
        o = tl.where(offsets_s >= i, o + s[:, i], o)

    tl.store(o_block_ptr, o, mask=mask)

@triton.jit
def chunk_gated_abc_fwd_kernel_h(
    k_ptr, v_ptr, g_ptr, h_ptr, h0_ptr, ht_ptr, T, K, V, BT, BK, BV, stride_k_T, stride_k_K, stride_v_T, stride_v_V, stride_g_T, stride_h_T, stride_h_V,
    BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_V: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_T
    offsets_t = block_start + tl.arange(0, BLOCK_SIZE_T)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    offsets_v = tl.arange(0, BLOCK_SIZE_V)
    mask = (offsets_t < T)[:, None, None] & (offsets_k < K)[None, :, None] & (offsets_v < V)[None, None, :]

    k_block_ptr = tl.make_block_ptr(
        base=k_ptr, shape=(T, K), strides=(stride_k_T, stride_k_K),
        offsets=(block_start, 0), block_shape=(BLOCK_SIZE_T, BLOCK_SIZE_K), order=(1, 0)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr, shape=(T, V), strides=(stride_v_T, stride_v_V),
        offsets=(block_start, 0), block_shape=(BLOCK_SIZE_T, BLOCK_SIZE_V), order=(1, 0)
    )
    g_block_ptr = tl.make_block_ptr(
        base=g_ptr, shape=(T,), strides=(stride_g_T,),
        offsets=(block_start, 0), block_shape=(BLOCK_SIZE_T, 1), order=(0,)
    )
    h_block_ptr = tl.make_block_ptr(
        base=h_ptr, shape=(T, V), strides=(stride_h_T, stride_h_V),
        offsets=(block_start, 0), block_shape=(BLOCK_SIZE_T, BLOCK_SIZE_V), order=(1, 0)
    )

    k = tl.load(k_block_ptr, mask=mask, other=0.0)
    v = tl.load(v_block_ptr, mask=mask, other=0.0)
    g = tl.load(g_block_ptr, mask=mask, other=0.0)

    b_h = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_V), dtype=tl.float32)
    if h0_ptr is not None:
        h0_block_ptr = tl.make_block_ptr(
            base=h0_ptr, shape=(1, V), strides=(1, stride_h_V),
            offsets=(0, 0), block_shape=(1, BLOCK_SIZE_V), order=(0, 1)
        )
        b_h = tl.load(h0_block_ptr, mask=(offsets_v < V)[None, :], other=0.0)

    for i in range(BLOCK_SIZE_T):
        b_k = k[i, :]
        b_v = v[i, :]
        b_g = g[i, 0]
        b_h = b_h * (1 - b_g) + b_k * b_v

    tl.store(h_block_ptr, b_h, mask=mask)

    if ht_ptr is not None:
        ht_block_ptr = tl.make_block_ptr(
            base=ht_ptr, shape=(1, V), strides=(1, stride_h_V),
            offsets=(0, 0), block_shape=(1, BLOCK_SIZE_V), order=(0, 1)
        )
        tl.store(ht_block_ptr, b_h[-1, :], mask=(offsets_v < V)[None, :], other=0.0)

### Wrapper Functions

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_T': 128, 'BLOCK_SIZE_S': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_T': 256, 'BLOCK_SIZE_S': 32}, num_warps=8),
    ],
    key=['T', 'S']
)
@triton.jit
def fwd_pre(s_ptr, o_ptr, T, S, stride_s_T, stride_s_S, stride_o_T, stride_o_S):
    chunk_gated_abc_fwd_kernel_cum[
        (T + 128 - 1) // 128,
    ](s_ptr, o_ptr, T, S, 128, 32, stride_s_T, stride_s_S, stride_o_T, stride_o_S, BLOCK_SIZE_T=128, BLOCK_SIZE_S=32)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_T': 128, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_V': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_T': 256, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_V': 32}, num_warps=8),
    ],
    key=['T', 'K', 'V']
)
@triton.jit
def fwd_inner(k_ptr, v_ptr, g_ptr, h_ptr, h0_ptr, ht_ptr, T, K, V, stride_k_T, stride_k_K, stride_v_T, stride_v_V, stride_g_T, stride_h_T, stride_h_V):
    chunk_gated_abc_fwd_kernel_h[
        (T + 128 - 1) // 128,
    ](k_ptr, v_ptr, g_ptr, h_ptr, h0_ptr, ht_ptr, T, K, V, 128, 32, 32, stride_k_T, stride_k_K, stride_v_T, stride_v_V, stride_g_T, stride_h_T, stride_h_V, BLOCK_SIZE_T=128, BLOCK_SIZE_K=32, BLOCK_SIZE_V=32)
