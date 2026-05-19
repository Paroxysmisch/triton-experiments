BLOCK_SIZE    = BLOCK_SIZE,
            num_warps     = num_warps,
        )
        ctx.save_for_backward(Q, cos, sin)
        return Q.view(batch, seq_len, n_heads, head_dim)

    @staticmethod
    def backward(ctx, DO):
        Q, cos, sin = ctx.saved_tensors
        batch, seq_len, n_heads, head_dim = Q.shape

        BLOCK_SIZE, num_warps = calculate_settings(head_dim//2)
        div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
        n_groups = div + (mod != 0)

        _rope_embedding[(n_rows, n_groups, )](
              Q,   Q.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len,
            head_dim, n_heads,
            BACKWARD_PASS = True,
            BLOCK_SIZE    = BLOCK_SIZE,
            num_warps     = num_warps,
        )
        return Q.view(batch, seq_len, n_heads, head_dim), None, None

def apply_rope_embedding(Q, cos, sin):
    return Fast_RoPE_Embedding.apply(Q, cos, sin)
