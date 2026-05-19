import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Tuple
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 256}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 512}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=5, num_warps=2),
    ],
    key=['block_size', 'hdim'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10
    },
)
@triton.jit
def _triton_rope(
    q_ptr, q_batch_stride, q_start_stride, q_head_stride, q_feature_stride,
    k_ptr, k_batch_stride, k_start_stride, k_head_stride, k_feature_stride,
    cos_ptr, cos_batch_stride, cos_start_stride, cos_head_stride,
    sin_ptr, sin_batch_stride, sin_start_stride, sin_head_stride,
    q_seq_len,
    cos_seq_len,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_seq_index = pid // tl.cdiv(q_seq_len, BLOCK_M)
    batch_index = batch_seq_index // q_seq_len
    seq_index = batch_seq_index % q_seq_len
    head_index = tl.program_id(axis=1)

    batch_strides = tl.load(q_batch_stride + batch_index)
    q_start_strides = tl.load(q_start_stride + batch_index)
    k_start_strides = tl.load(k_start_stride + batch_index)
    head_strides = tl.load(q_head_stride + head_index)
    feature_strides = tl.load(q_feature_stride + head_index)

    batch_pointer = batch_strides + batch_seq_index * BLOCK_M
    q_start_pointer = q_start_strides + seq_index * BLOCK_M
    k_start_pointer = k_start_strides + seq_index * BLOCK_N
    head_pointer = head_strides
    feature_pointer = feature_strides

    cos_pointer = tl.load(cos_ptr) + tl.load(cos_batch_stride + batch_index) + tl.load(cos_start_stride + batch_index) + head_strides
    sin_pointer = tl.load(sin_ptr) + tl.load(sin_batch_stride + batch_index) + tl.load(sin_start_stride + batch_index) + head_strides

    q_ptr_mask = (batch_pointer + tl.arange(0, BLOCK_M)) < q_seq_len
    k_ptr_mask = (k_start_pointer + tl.arange(0, BLOCK_N)) < q_seq_len

    q_ptr += q_ptr_mask * feature_pointer * q_feature_stride + q_start_pointer + feature_pointer * q_feature_stride
    k_ptr += k_ptr_mask * feature_pointer * k_feature_stride + k_start_pointer + feature_pointer * k_feature_stride

    cos_pointer += tl.arange(0, BLOCK_N)
    sin_pointer += tl.arange(0, BLOCK_N)

    cos_mask = (tl.arange(0, BLOCK_N) < HEAD_SIZE)
    cos_pointer += cos_mask * head_pointer
    sin_pointer += cos_mask * head_pointer

    q1 = tl.load(q_ptr).to(tl.float32)
    q2 = tl.load(q_ptr + feature_strides).to(tl.float32)
    if BACKWARD_PASS:
        cos_theta = -tl.load(cos_pointer).to(tl.float32)
        sin_theta = -tl.load(sin_pointer).to(tl.float32)
    else:
        cos_theta = tl.load(cos_pointer).to(tl.float32)
        sin_theta = tl.load(sin_pointer).to(tl.float32)

    q1_new = cos_theta * q1 - sin_theta * q2
    q2_new = sin_theta * q1 + cos_theta * q2

    tl.store(q_ptr, q1_new.to(q_ptr.dtype.element_ty))
    tl.store(q_ptr + feature_strides, q2_new.to(q_ptr.dtype.element_ty))

def rope_forward(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, max_seq_len: int, backward_pass: bool = False
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    transposed_q = q.transpose(1, 2).clone()
    transposed_k = k.transpose(1, 2).clone()
    original_q_shape = transposed_q.shape
    original_k_shape = transposed_k.shape
    transposed_q = transposed_q.reshape(-1, transposed_q.shape[-1])
    transposed_k = transposed_k.reshape(-1, transposed_k.shape[-1])
    padded_q, q_batch_stride, q_start_stride, q_head_stride, q_feature_stride = triton_util.contiguous_strides(
        transposed_q
    )
    padded_k, k_batch_stride, k_start_stride, k_head_stride, k_feature_stride = triton_util.contiguous_strides(
        transposed_k
    )
    if backward_pass:
        cos, _ = cos.reverse(0).contiguous()
        sin, _ = sin.reverse(0).cont
