cos.squeeze(), sin.squeeze()
        Q = Q.contiguous()

        seqlen, n_heads, head_dim = Q.shape[-3:]
        assert Q.shape[-2] == cos.shape[0] and Q.shape[-2] == sin.shape[0], \
            "Q and sin/cos matrices must be of the same size along their sequence dim"

        n_elements = Q.numel()
        BLOCK_SIZE, num_warps = calculate_settings(head_dim)

        Q = Q.view(n_elements)
        q_embedded = torch.empty_like(Q)
        _rope_embedding[(Q.shape[0], Q.shape[1] // ROPE_GROUP_SIZE)](
            Q,     Q.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seqlen,
            head_dim, n_heads,
            BACKWARD_PASS = False,
            BLOCK_SIZE    = BLOCK_SIZE,
            num_warps     = num_warps
        )
        q_embedded = q_embedded.view(Q.shape)
        ctx.save_for_backward(q_embedded, cos, sin)
        return q_embedded

    @staticmethod
    def backward(ctx, A_grad_rope):
        q_embedded, cos, sin = ctx.saved_tensors

        seqlen, n_heads, head_dim = q_embedded.shape[-3:]
        assert A_grad_rope.shape[-2] == cos.shape[0] and A_grad_rope.shape[-2] == sin.shape[0], \
            "Grad and sin/cos matrices must be of the same size along their sequence dim"

        n_elements = A_grad_rope.numel()
        BLOCK_SIZE, num_warps = calculate_settings(head_dim)

        A_grad_rope = A_grad_rope.contiguous().view(n_elements)
        q_embedded = q_embedded.contiguous().view(n_elements)
        q_grad = torch.empty_like(A_grad_rope)

        _rope_embedding[(A_grad_rope.shape[0], A_grad_rope.shape[1] // ROPE_GROUP_SIZE)](
            q_embedded, q_embedded.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seqlen,
            head_dim, n_heads,
            BACKWARD_PASS = True,
            BLOCK_SIZE    = BLOCK_SIZE,
            num_warps     = num_warps
        )
        q_grad = q_grad.view(A_grad_rope.shape)
        return q_grad, None, None

def fast_rope_embedding(Q, K, cos, sin):
    Q = transpose_last_dim_for_RoPE(Q)
    K = transpose_last_dim_for_RoPE(K)

    q_embedded = Fast_RoPE_Embedding.apply(Q, cos, sin)
    k_embedded = Fast_RoPE_Embedding.apply(K, cos, sin)

    q_embedded = transpose_last_dim_back_from_RoPE(q_embedded)
    k_embedded = transpose_last_dim_back_from_RoPE(k_embedded)

    return q_embedded, k_embedded
