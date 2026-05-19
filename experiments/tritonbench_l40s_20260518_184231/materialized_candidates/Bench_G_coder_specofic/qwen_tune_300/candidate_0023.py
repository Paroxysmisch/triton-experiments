import torch
import triton
import triton.language as tl

@triton.jit
def rotary_embedding_kernel(
    q, k, cos, sin,  # q and k are in [batch, out_seq_len, num_head, head_dim] and cos, sin are in [num_head, head_dim]
    k_cache,
    seq_len,
    stride_b, stride_qo, stride_ql, stride_h, stride_hk, stride_hc, stride_hsin,  # stride of input tensor
    stride_kb, stride_ko, stride_kl,  # stride of key tensor
    stride_cos_h, stride_cos_d,  # stride of cos tensor
    stride_sin_h, stride_sin_d,  # stride of sin tensor
    stride_k_cache,  # stride of k_cache
    head_dim: tl.constexpr,
    Q_HEAD_NUM: tl.constexpr,
    PAST_MAX_LEN: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    num_warps: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    remainder_len = tl.program_id(2)

    cur_batch_seq_len = tl.load(seq_len + cur_batch)

    # compute block ptr
    batch_block_ptr = tl.make_block_ptr(
        base=q + cur_batch * stride_b,
        shape=(Q_HEAD_NUM, head_dim),
        strides=(stride_qo, stride_ql),
        offsets=(cur_head * stride_h, 0),
        block_shape=(head_dim, 1),
        order=(1, 0),
    )
    kv_block_ptr = tl.make_block_ptr(
        base=k + cur_batch * stride_kb,
        shape=(Q_HEAD_NUM, head_dim),
        strides=(stride_ko, stride_kl),
        offsets=(cur_head * stride_hk, 0),
        block_shape=(head_dim, 1),
        order=(1, 0),
    )
    cos_block_ptr = tl.make_block_ptr(
        base=cos,
        shape=(Q_HEAD_NUM, head_dim),
        strides=(stride_cos_h, stride_cos_d),
        offsets=(cur_head * stride_hc, 0),
        block_shape=(head_dim, 1),
        order=(1, 0),
    )
    sin_block_ptr = tl.make_block_ptr(
        base=sin,
        shape=(Q_HEAD_NUM, head_dim),
        strides=(stride_sin_h, stride_sin_d),
        offsets=(cur_head * stride_hsin, 0),
        block_shape=(head_dim, 1),
        order=(1, 0),
    )
    past_len = tl.maximum(0, remainder_len - cur_batch_seq_len)
    past_block_ptr = tl.make_block_ptr(
        base=k_cache + past_len * stride_k_cache,
        shape=(Q_HEAD_NUM, head_dim),
        strides=(stride_k_cache, 0),
        offsets=(cur_head, 0),
        block_shape=(1, head_dim),
        order=(0, 1),
    )

    past_seq_len = tl.minimum(remainder_len, cur_batch_seq_len)
    for i in range(0, past_seq_len, head_dim):
        # load q
        q = tl.load(
            batch_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        # load kv
        kv = tl.load(
            kv_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        # save to cache
        tl.store(
            past_block_ptr,
            kv,
            boundary_check=(0, 1),
        )
        # move block ptr
        batch_block_ptr = tl.advance(batch_block_ptr, (0, 1))
        kv_block_ptr = tl.advance(kv_block_ptr, (0, 1))
        past_block_ptr = tl.advance(past_block_ptr, (1, 0))

    if ROTARY_INTERLEAVED:
        batch_block_ptr = tl.make_block_ptr(
            base=q + cur_batch * stride_b,
            shape=(Q_HEAD_NUM, head_dim),
            strides=(stride_qo, stride_ql),
            offsets=(cur_head * stride_h, 0),
            block_shape=(head_dim, 1),
            order=(1, 0),
        )
        kv_block_ptr = tl.make_block_ptr(
            base=k + cur_batch * stride_kb,
            shape=(Q_HEAD_NUM, head_dim),
            strides=(stride_ko, stride_kl),
            offsets=(cur_head * stride_hk, 0),
            block_shape=(head_dim, 1),
            order=(1, 0),
        )
        cos_block_ptr = tl.make_block_ptr(
            base=cos,
            shape=(Q_HEAD_NUM, head_dim),
            strides=(stride_cos_h, stride_cos_d),
            offsets=(cur_head * stride_hc, 0),
            block_shape=(head_dim, 1),
            order=(1, 0),
        )
        sin_block_ptr = tl.make_block_ptr(
            base=sin,
            shape=(Q_HEAD_NUM, head_dim),
            strides=(stride_sin_h, stride_sin_d),
            offsets=(cur_head * stride_hsin, 0),
            block_shape=(head_dim, 1),
            order=(1, 0),
        )
        for i in range(0, cur_batch_seq_len, head_dim):
            # load q
            loaded_q = tl.load(
                batch_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            # load kv
            loaded_kv = tl.load(
                kv_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            # load cos sin
            loaded_cos = tl.load(cos_block_ptr, boundary_check=(0, 1))
            loaded_sin = tl.load(sin_block_ptr, boundary_check=(0, 1))
            # apply rotary
            rotated_q0 = tl.where(
                tl.arange(0, 1) < 1,
                loaded_q * loaded_cos - loaded_kv * loaded_sin,
                0,
            )
            rotated_q1 = tl.where(
                tl.arange(0, 1) < 1,
                loaded_q * loaded_sin + loaded_kv * loaded_cos,
                0,
            )
            # save result
            tl.store(
                batch_block_ptr,
                rotated_q0,
                boundary_check=(0, 1),
            )
            tl.store(
                kv_block_ptr,
                rotated_q1,
                boundary_check=(0, 1),
            )
            # move block ptr
            batch_block_ptr = tl.advance(batch_block_ptr, (0, 1))
            kv_block_ptr = tl.advance(kv_block_ptr, (0, 1))
            cos_block_ptr = tl.advance(cos_block_ptr, (0, 1))
            sin_block_ptr = tl.advance(sin_block_ptr, (0, 1))
    else:
        batch_block_ptr = tl.make_block_ptr(
            base=q + cur_batch * stride_b,
            shape=(Q_HEAD_NUM, head_dim),
            strides=(stride_qo, stride_ql),
            offsets=(cur_head * stride_h, 0),
            block_shape=(1, head_dim),
            order=(1, 0),
        )
        kv_block_ptr = tl.make_block_ptr(
            base=k + cur_batch * stride_kb,
            shape=(Q_HEAD_NUM, head_dim),
            strides=(stride_ko, stride_kl),
            offsets=(cur_head * stride_hk, 0),
            block_shape=(1, head_dim),
            order=(1, 0),
        )
        cos_block_ptr = tl.make_block_ptr(
            base=cos,
            shape=(Q_HEAD_NUM, head_dim),
            strides=(stride_cos_h, stride_cos_d),
            offsets=(cur_head * stride_hc, 0),
            block_shape=(1, head_dim),
            order=(1, 0),
        )
        sin_block_ptr = tl.make_block_ptr(
            base=sin,
            shape=(Q_HEAD_NUM, head_dim),
            strides=(stride_sin_h, stride_sin_d),
            offsets=(cur_head * stride_hsin, 0),
            block_shape=(1, head_dim),
            order=(1, 0),
        )
        for i in range(0, cur_batch_seq_len, head_dim):
            # load q
            loaded_q = tl.load(
                batch_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            # load kv
            loaded_kv = tl.load(
                kv_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            # load cos sin
            loaded_cos = tl.load(cos_block_ptr, boundary_check=(0, 1))
            loaded_sin = tl.load(sin_block_ptr, boundary_check=(0, 1))
            # apply rotary
            rotated_q = loaded_q * loaded_cos - loaded_kv * loaded_sin
            # save result
            tl.store(
                batch_block_ptr,
                rotated_q,
                boundary_check=(0, 1),
            )
            # move block ptr
            batch_block_ptr = tl.advance(batch_block_ptr, (1, 0))
            kv_block_ptr = tl.advance(kv_block_ptr, (1, 0))
            cos_block_ptr = tl.advance(cos_block_ptr, (1, 0))
            sin_block_ptr = tl.advance(sin_block_ptr, (1, 0))

@torch.inference_mode()
def fused_rotary_embedding_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor = None,
    block_tables: torch.Tensor = None
