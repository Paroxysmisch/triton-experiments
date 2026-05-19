import torch
import triton
import triton.language as tl


@triton.jit
def rotary_embedding_kernel(
    q, k, cos, sin,
    Q_HEAD_NUM: tl.constexpr, HEAD_DIM: tl.constexpr,
    stride_q0: tl.constexpr, stride_q1: tl.constexpr,
    stride_k0: tl.constexpr, stride_k1: tl.constexpr,
    stride_cos: tl.constexpr, stride_sin: tl.constexpr,
    K_HEAD_NUM: tl.constexpr,
    BLOCK: tl.constexpr,
    k_cache: tl.constexpr, block_tables: tl.constexpr, kv_lengths: tl.constexpr,
    rotary_dim: tl.constexpr,
    num_warps: tl.constexpr,
):
    # rotary positional embedding for transformer
    cur_head_dim = HEAD_DIM // 2
    cur_head_dim_2 = cur_head_dim * 2
    head_idx = tl.program_id(0) // Q_HEAD_NUM
    cur_head_idx = head_idx % CUR_HEAD_NUM
    seq_idx = tl.program_id(0) % Q_HEAD_NUM
    cur_head_start_idx = cur_head_idx * cur_head_dim
    cur_head_end_idx = cur_head_start_idx + cur_head_dim

    cur_head_start_idx_2 = cur_head_idx * cur_head_dim_2
    cur_head_end_idx_2 = cur_head_start_idx_2 + cur_head_dim_2

    loaded_cos = tl.load(cos + seq_idx * stride_cos + cur_head_start_idx_2)
    loaded_sin = tl.load(sin + seq_idx * stride_sin + cur_head_start_idx_2)

    loaded_q0 = tl.load(q + seq_idx * stride_q1 + cur_head_start_idx + cur_head_dim:
                        loaded_q0.dtype.element_ty,
                        mask=cur_head_start_idx + cur_head_dim < cur_head_end_idx,
                        other=0.0)
    loaded_q1 = tl.load(q + seq_idx * stride_q1 + cur_head_start_idx:loaded_q0.dtype.element_ty,
                        mask=cur_head_start_idx < cur_head_end_idx,
                        other=0.0)

    out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
    out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos

    if k is not None:
        loaded_k0 = tl.load(k + seq_idx * stride_k1 + cur_head_start_idx + cur_head_dim:
                            loaded_k0.dtype.element_ty,
                            mask=cur_head_start_idx + cur_head_dim < cur_head_end_idx,
                            other=0.0)
        loaded_k1 = tl.load(k + seq_idx * stride_k1 + cur_head_start_idx:loaded_k0.dtype.element_ty,
                            mask=cur_head_start_idx < cur_head_end_idx,
                            other=0.0)

        out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
        out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos

    if k_cache is None:
        tl.store(q + seq_idx * stride_q1 + cur_head_start_idx:loaded_q0.dtype.element_ty,
                 out_q0,
                 mask=cur_head_start_idx < cur_head_end_idx)
        tl.store(q + seq_idx * stride_q1 + cur_head_start_idx + cur_head_dim:loaded_q0.dtype.element_ty,
                 out_q1,
                 mask=cur_head_start_idx + cur_head_dim < cur_head_end_idx)
        if k is not None:
            tl.store(k + seq_idx * stride_k1 + cur_head_start_idx:loaded_k0.dtype.element_ty,
                     out_k0,
                     mask=cur_head_start_idx < cur_head_end_idx)
            tl.store(k + seq_idx * stride_k1 + cur_head_start_idx + cur_head_dim:loaded_k0.dtype.element_ty,
                     out_k1,
                     mask=cur_head_start_idx + cur_head_dim < cur_head_end_idx)
    else:
        cur_head_idx = head_idx % K_HEAD_NUM
        cur_head_start_idx = cur_head_idx * cur_head_dim_2
        cur_head_end_idx = cur_head_start_idx + cur_head_dim_2

        loaded_kv_length = tl.load(kv_lengths + seq_idx)
        loaded_block_idx = tl.load(block_tables + seq_idx)

        store_idx = (loaded_block_idx * BLOCK + cur_head_start_idx_2 // 2) * stride_k0 + cur_head_start_idx_2
        store_mask = (loaded_block_idx * BLOCK + cur_head_start_idx_2 // 2) < (loaded_kv_length + BLOCK)
        store_mask = store_mask & (cur_head_start_idx_2 < cur_head_end_idx_2)

        tl.store(k_cache + store_idx,
                 out_q0,
                 mask=store_mask)
        tl.store(k_cache + store_idx + 1,
                 out_q1,
                 mask=store_mask)

        if k is not None:
            store_idx = (loaded_block_idx * BLOCK + cur_head_start_idx_2 // 2) * stride_k0 + cur_head_start_idx_2 + 1
            tl.store(k_cache + store_idx,
                     out_k0,
                     mask=store_mask)
            tl.store(k_cache + store_idx + 1,
                     out_k1,
                     mask=store_mask)


@triton.jit
def fused_rotary_embedding_kernel_v2(
    q, k, cos, sin,
    k_cache, block_tables, kv_lengths,
    Q_HEAD_NUM: tl.constexpr, HEAD_DIM: tl.constexpr,
    stride_q0: tl.constexpr, stride_q1: tl.constexpr,
    stride_k0: tl.constexpr, stride_k1: tl.constexpr,
    stride_cos: tl.constexpr, stride_sin: tl.constexpr,
    K_HEAD_NUM: tl.constexpr,
    BLOCK: tl.constexpr,
    rotary_dim: tl.constexpr,
    num_warps: tl.constexpr,
):
    cur_head_dim = HEAD_DIM // 2
    cur_head_dim_2 = cur_head_dim * 2
    head_idx = tl.program_id(0) // Q_HEAD_NUM
    cur_head_idx = head_idx % CUR_HEAD_NUM
    seq_idx = tl.program_id(0) % Q_HEAD_NUM
    cur_head_start_idx = cur_head_idx * cur_head_dim
    cur_head_end_idx = cur_head_start_idx + cur_head_dim

    cur_head_start_idx_2 = cur_head_idx * cur_head_dim_2
    cur_head_end_idx_2 = cur_
