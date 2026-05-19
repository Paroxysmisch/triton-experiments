.load(Q + offs_q2, mask = mask, other = 0).to(sin1.dtype)

        tl.store(Q + offs_q1, Q1*cos1 - Q2*sin1, mask = mask)
        tl.store(Q + offs_q2, Q2*cos1 + Q1*sin1, mask = mask)
        pass
    pass

def _rope_embedding_forward_impl(Q, cos, sin):
    seqlen        = Q.shape[0]
    head_dim      = Q.shape[2]
    n_heads       = Q.shape[1]

    Q = Q.view(-1, n_heads*head_dim)
    Q = Q.transpose(0, 1)

    n_rows = Q.shape[0]
    n_cols = Q.shape[1]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    with torch.cuda.device(Q.device.index):
        _rope_embedding[(n_rows, n_heads//ROPE_GROUP_SIZE, )](
            Q,     Q.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seqlen,
            head_dim, n_heads,
            BACKWARD_PASS = False,
            BLOCK_SIZE    = BLOCK_SIZE,
            num_warps     = num_warps,
        )
    pass

def _rope_embedding_backward_impl(dY, cos, sin, n_groups, BLOCK_SIZE, num_warps):
    seqlen        = dY.shape[0]
    head_dim      = dY.shape[2]
    n_heads       = dY.shape[1]

    dY = dY.view(-1, n_heads*head_dim)
    dY = dY.transpose(0, 1)

    n_rows = dY.shape[0]
    n_cols = dY.shape[1]

    with torch.cuda.device(dY.device.index):
        _rope_embedding[(n_rows, n_groups, )](
            dY,     dY.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seqlen,
            head_dim, n_heads,
            BACKWARD_PASS = True,
            BLOCK_SIZE    = BLOCK_SIZE,
            num_warps     = num_warps,
        )
    pass
