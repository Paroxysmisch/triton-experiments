prog = _attn_fwd.compile(grid=(grid,), inputs=(q, k, v, q_scale, k_scale, ... ))
triton.testing.run(prog, inputs=(q, k, v, q_scale, k_scale, ... ))
