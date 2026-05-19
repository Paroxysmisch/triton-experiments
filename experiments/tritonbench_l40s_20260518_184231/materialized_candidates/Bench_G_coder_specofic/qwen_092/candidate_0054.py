import triton
import triton.language as tl

# Define the wrapper for cumulative sum
def fwd_pre(g, T, S, BT, BS):
    g = g.reshape(T, S)
    grid = (T + BT - 1) // BT
    s = triton.empty((grid, T, S), dtype=g.dtype)
    o = triton.empty((grid, T, S), dtype=g.dtype)
    triton.kernel(chunk_gated_abc_fwd_kernel_cum, grid=(grid,), block=(BS,), inps=[g, s, T, S, BT, BS], outs=[o])
    return o

# Define the wrapper for gated cumulative sum
def fwd_inner(k, v, g, h0, ht, h, T, K, V, BT, BK, BV):
    grid = (T + BT - 1) // BT
    triton.kernel(chunk_gated_abc_fwd_kernel_h, grid=(grid,), block=(BK,), inps=[k, v, g, h0, ht, h, T, K, V, BT, BK, BV], outs=[h])
