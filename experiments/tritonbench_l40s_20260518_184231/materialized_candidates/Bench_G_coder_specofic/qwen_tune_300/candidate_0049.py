import torch
import triton
import triton.language as tl

@triton.jit
def _fill_kv_cache_kernel(
    k_states,
    v_states,
    k_caches,
    v_caches,
    q_start_loc,
    seq_lens,
    k_scale,
    v_scale,
    k_scales_zeros,
    v_scales_zeros,
    cur_batch,
    cur_head,
    max_num_blocks,
    stride_b_h,
    stride_h_d,
    stride_h_n,
    stride_k_cache_h,
    stride_k_cache_d,
    stride_k_cache_b,
    stride_v_cache_h,
    stride_v_cache_d,
    stride_v_cache_b,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch_seq_len = tl.load(seq_lens + cur_batch)
    cur_batch_start_index = tl.load(q_start_loc + cur_batch)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    off_k_cache = (
        (cur_head * stride_h_d + offs_d[:, None]) * stride_k_cache_d
        + (cur_batch_start_index + offs_n[None, :]) * stride_k_cache_b
    )
    off_v_cache = (
        cur_head * stride_h_d + offs_d[:, None]
    ) * stride_v_cache_d + (cur_batch_start_index + offs_n[None, :]) * stride_v_cache_b

    off_k_states = cur_head * stride_h_d + offs_d[:, None] + offs_n[None, :] * stride_h_n
    off_v_states = cur_head * stride_h_d + offs_d[:, None] + offs_n[None, :] * stride_h_n

    src_mask = (cur_batch_start_index + offs_n[None, :]) < cur_batch_seq_len
    k_states_ptrs = k_states + off_k_states
    v_states_ptrs = v_states + off_v_states

    k_cache_ptrs = k_caches + off_k_cache
    v_cache_ptrs = v_caches + off_v_cache

    k_scale_ptrs = k_scale + cur_head
    v_scale_ptrs = v_scale + cur_head

    k_scales_zeros_ptrs = k_scales_zeros + cur_head
    v_scales_zeros_ptrs = v_scales_zeros + cur_head

    for i in range(0, max_num_blocks, 1):
        src_index = cur_batch_start_index + offs_n
        cur_batch_src_mask = src_index < cur_batch_seq_len

        k_states_cur = tl.load(k_states_ptrs, mask=src_mask & cur_batch_src_mask, other=0)
        v_states_cur = tl.load(v_states_ptrs, mask=src_mask & cur_batch_src_mask, other=0)

        tl.store(k_cache_ptrs, k_states_cur, mask=cur_batch_src_mask)
        tl.store(v_cache_ptrs, v_states_cur, mask=cur_batch_src_mask)

        k_states_ptrs += stride_b_h
        v_states_ptrs += stride_b_h

        k_cache_ptrs += stride_k_cache_h
        v_cache_ptrs += stride_v_cache_h

def fill_kv_cache(
    k_states: torch.Tensor,
    v_states: torch.Tensor,
    k_caches: torch.Tensor,
    v_caches: torch.Tensor,
    q_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    k_scales_zeros: torch.Tensor = None,
    v_scales_zeros: torch.Tensor = None,
    quant_policy: int = 0,
):
    if quant_policy == 0:
        _fill_kv_cache_kernel[
            lambda meta: (
                k_states.shape[0],
                triton.cdiv(v_states.shape[1], meta["BLOCK_N"]),
            )
        ](
            k_states,
            v_states,
            k_caches,
            v_caches,
            q_start_loc,
            seq_lens,
            k_scale,
            v_scale,
            k_scales_zeros,
            v_scales_zeros,
            0,
            0,
            k_states.shape[1],
            k_states.stride(0),
            k_states.stride(1),
            k_states.stride(2),
            k_caches.stride(0),
            k_caches.stride(1),
            k_caches.stride(2),
            v_caches.stride(0),
            v_caches.stride(1),
            v_caches.stride(2),
            BLOCK_DMODEL=k_states.shape[1],
            BLOCK_N=triton.next_power_of_2(v_states.shape[1]),
        )
    else:
        _fill_kv_cache_quant_kernel[
            lambda meta: (
                k_states.shape[0],
                triton.cdiv(v_states.shape[1], meta["BLOCK_N"]),
            )
        ](
            k_states,
            v_states,
            k_caches,
            v_caches,
            q_start_loc,
            seq_lens,
            k_scale,
            v_scale,
            k_scales_zeros,
            v_scales_zeros,
            0,
            0,
            k_states.shape[1],
            k_states.stride(0),
            k_states.stride(1),
            k_states.stride(2),
            k_caches.stride(0),
            k_caches.stride(1),
            k_caches.stride(2),
            v_caches.stride(0),
            v_caches.stride(1),
            v_caches.stride(2),
            quant_policy,
            BLOCK_DMODEL=k_states.shape[1],
            BLOCK_N=triton.next_power_of_2(v_states.shape[1]),
        )
