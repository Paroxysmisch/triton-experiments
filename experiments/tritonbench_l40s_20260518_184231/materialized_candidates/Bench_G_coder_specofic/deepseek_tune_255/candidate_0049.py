import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Tuple

@triton.jit
def _fill_kv_cache_kernel(
    k_states, v_states, k_caches, v_caches,
    q_start_loc, seq_len,
    batch_size, max_num_blocks,
    stride_k_bs, stride_k_h, stride_k_bl, stride_k_d,
    stride_v_bs, stride_v_h, stride_v_bl, stride_v_d,
    stride_q_start_loc_b, stride_q_start_loc_s,
    BLOCK_HEAD: tl.constexpr, BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    quant_policy: tl.constexpr,
    k_scales_zeros: Optional[Tensor], v_scales_zeros: Optional[Tensor],
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_len_cur_batch = tl.load(seq_len + cur_batch)
    kv_cache_cur_batch_head_start = k_caches + cur_batch * stride_k_bs + cur_head * stride_k_h
    q_start_loc_cur_batch = q_start_loc + cur_batch * stride_q_start_loc_b

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)
    blk_idx_x = tl.arange(0, BLOCK_SEQ)
    mask_t = blk_idx_x[:, None] < seq_len_cur_batch
    kv_states_cur_batch_head_offs = cur_head * stride_k_bl + offs_d[None, :]
    kv_cache_cur_batch_head_offs = cur_head * stride_k_h + offs_d[None, :]
    q_start_loc_cur_batch_offs = cur_head * stride_q_start_loc_s

    k_states_cur_batch = k_states + cur_batch * stride_k_bs
    v_states_cur_batch = v_states + cur_batch * stride_v_bs

    for blk_idx_y in range(0, tl.cdiv(seq_len_cur_batch, BLOCK_SEQ)):
        kv_cache_cur_batch_head = kv_cache_cur_batch_head_start + blk_idx_y * stride_k_bl
        kv_states_cur_batch_head = kv_states_cur_batch + kv_states_cur_batch_head_offs
        q_start_loc_cur_batch_head = q_start_loc_cur_batch + q_start_loc_cur_batch_offs

        k_state = tl.load(kv_states_cur_batch_head, mask=mask_t, other=0.0)
        v_state = tl.load(kv_states_cur_batch_head + stride_k_d, mask=mask_t, other=0.0)
        q_start_loc = tl.load(q_start_loc_cur_batch_head, mask=mask_t, other=0)

        if quant_policy == 0:
            tl.store(kv_cache_cur_batch_head + kv_cache_cur_batch_head_offs, k_state, mask=mask_t)
            tl.store(kv_cache_cur_batch_head + stride_k_d + kv_cache_cur_batch_head_offs, v_state, mask=mask_t)
        tl.store(q_start_loc_cur_batch_head + cur_head, q_start_loc, mask=mask_t)


@triton.jit
def _fill_kv_cache_quant_kernel(
    k_states, v_states, k_caches, v_caches,
    q_start_loc, seq_len,
    batch_size, max_num_blocks,
    stride_k_bs, stride_k_h, stride_k_bl, stride_k_d,
    stride_v_bs, stride_v_h, stride_v_bl, stride_v_d,
    stride_q_start_loc_b, stride_q_start_loc_s,
    k_scales_zeros, v_scales_zeros,
    BLOCK_HEAD: tl.constexpr, BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    k_hist_table: tl.constexpr, v_hist_table: tl.constexpr,
    k_quant_type: tl.constexpr, v_quant_type: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_len_cur_batch = tl.load(seq_len + cur_batch)
    kv_cache_cur_batch_head_start = k_caches + cur_batch * stride_k_bs + cur_head * stride_k_h
    q_start_loc_cur_batch = q_start_loc + cur_batch * stride_q_start_loc_b

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)
    blk_idx_x = tl.arange(0, BLOCK_SEQ)
    mask_t = blk_idx_x[:, None] < seq_len_cur_batch
    kv_states_cur_batch_head_offs = cur_head * stride_k_bl + offs_d[None, :]
    kv_cache_cur_batch_head_offs = cur_head * stride_k_h + offs_d[None, :]
    q_start_loc_cur_batch_offs = cur_head * stride_q_start_loc_s

    k_states_cur_batch = k_states + cur_batch * stride_k_bs
    v_states_cur_batch = v_states + cur_batch * stride_v_bs

    k_scales, k_zeros = tl.load(k_scales_zeros + cur_head * 2)
    v_scales, v_zeros = tl.load(v_scales_zeros + cur_head * 2)

    for blk_idx_y in range(0, tl.cdiv(seq_len_cur_batch, BLOCK_SEQ)):
        kv_cache_cur_batch_head = kv_cache_cur_batch_head_start + blk_idx_y * stride_k_bl
        kv_states_cur_batch_head = kv_states_cur_batch + kv_states_cur_batch_head_offs
        q_start_loc_cur_batch_head = q_start_loc_cur_batch + q_start_loc_cur_batch_offs

        k_state = tl.load(kv_states_cur_batch_head, mask=mask_t, other=0.0)
        v_state = tl.load(kv_states_cur_batch_head + stride_k_d, mask=mask_t, other=0.0)
        q_start_loc = tl.load(q_start_loc_cur_batch_head, mask=mask_t, other=0)

        k_state_quant = _quant_func_map[k_quant_type](k_state, k_scales, k_zeros, k_hist_table)
        v_state_quant = _quant_func_map[
