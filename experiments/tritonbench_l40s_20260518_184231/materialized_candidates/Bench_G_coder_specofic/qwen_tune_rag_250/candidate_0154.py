KV_GROUP_NUM=4,
            HEAD_DIM=head_dim,
        )
    else:
        assert head_dim <= 128
        decoding_fused_rotary_embedding(
            q,
            k,
            v=None,
            cos=cos,
            sin=sin,
            k_cache=k_cache,
            block_tables=block_tables,
            kv_lengths=kv_lengths,
            use_new_kcache_layout=True,
        )

def fused_rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    use_new_kcache_layout: bool = False,
):
    # Host function for fused rotary embedding of queries, keys, and cache update

def decoding_fused_rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    v: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    kv_lengths: Optional[torch.Tensor] = None,
    use_new_kcache_layout: bool = False,
):
    # Host function for decoding stage's fused rotary embedding
