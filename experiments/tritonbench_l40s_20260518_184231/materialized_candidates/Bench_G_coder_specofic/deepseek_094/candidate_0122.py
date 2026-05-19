import triton.language.c as c

@c.jit
def Fast_RoPE_Embedding(Q, Q_row_stride, cos, sin, cos_row_stride, sin_row_stride, seqlen, head_dim, n_heads, BACKWARD_PASS, BLOCK_SIZE):
    grid = lambda : (c.ceil(seqlen / BLOCK_SIZE), 1, 1)
    _rope_embedding[grid](Q, Q_row_stride, cos, sin, cos_row_stride, sin_row_stride, seqlen, head_dim, n_heads, BACKWARD_PASS, BLOCK_SIZE)
