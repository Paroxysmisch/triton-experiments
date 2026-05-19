import math
import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd
from torch import Tensor
from typing import Optional

MAX_FUSED_SIZE = 65536 // 4  # max number of classes

def calculate_settings(n):
    assert n <= MAX_FUSED_SIZE, f"label size {n} exceeds MAX_FUSED_SIZE"
    num_warps = 4
    while n > 1024:
        n //= 2
        num_warps *= 2
    return num_warps, min(n, 2048)

@triton.jit
def _cross_entropy_forward(
    input_row,
    logit_scaling,
    logits_ptr,
    logits_row_stride,
    labels_ptr,
    label_ignore_value,
    loss_ptr,
    logit_cap: tl.constexpr = 20.0,
    logit_shift: tl.constexpr = 0.0,
    logit_scale: tl.constexpr = 1.0,
    fused_size: tl.constexpr = 0,
    logits_max_ptr: tl.constexpr = None,
    logits_min_ptr: tl.constexpr = None,
    row_idx: tl.constexpr = 0,
    BLOCK_SIZE: tl.constexpr = 1,
):
    # Triton kernel to compute cross-entropy loss for a single row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptr = input_row + col_offsets
    logits_row_offsets = row_idx * logits_row_stride + col_offsets
    logits_mask = col_offsets < fused_size
    logits_row_ptr = logits_ptr + logits_row_offsets
    logits = tl.load(logits_row_ptr, mask=logits_mask, other=0.0).to(tl.float32)
    if logit_scaling:
        logits = (logits * logit_scale) + logit_shift
    if logit_cap is not None:
        logits = tl.minimum(tl.maximum(logits, -logit_cap), logit_cap)
    if logits_max_ptr is not None and logits_min_ptr is not None:
        logits_max = tl.load(logits_max_ptr)
        logits_min = tl.load(logits_min_ptr)
        logits = (logits - logits_min) / (logits_max - logits_min)
    label_value = tl.load(labels_ptr + row_idx)
    label_ignore_mask = label_value != label_ignore_value
    logits = tl.where(label_ignore_mask, logits, float("-inf"))
    loss = -tl.log(tl.sum(tl.exp(logits)))
    tl.store(loss_ptr + row_idx, loss)

@triton.jit
def _chunked_cross_entropy_forward(
    input_row,
    logit_scaling,
    logits_ptr,
    logits_row_stride,
    labels_ptr,
    label_ignore_value,
    loss_ptr,
    chunk_size_ptr,
    chunk_start_ptr,
    logit_cap: tl.constexpr = 20.0,
    logit_shift: tl.constexpr = 0.0,
    logit_scale: tl.constexpr = 1.0,
    fused_size: tl.constexpr = 0,
    logits_max_ptr: tl.constexpr = None,
    logits_min_ptr: tl.constexpr = None,
    row_idx: tl.constexpr = 0,
    BLOCK_SIZE: tl.constexpr = 1,
):
    # Triton kernel to compute cross-entropy loss for a single row with chunking
    chunk_size = tl.load(chunk_size_ptr)
    chunk_start = tl.load(chunk_start_ptr)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptr = input_row + col_offsets
    logits_row_offsets = (row_idx - chunk_start) * logits_row_stride + col_offsets
    logits_mask = col_offsets < fused_size
    logits_row_ptr = logits_ptr + logits_row_offsets
    logits = tl.load(logits_row_ptr, mask=logits_mask, other=0.0).to(tl.float32)
    if logit_scaling:
        logits = (logits * logit_scale) + logit_shift
    if logit_cap is not None:
        logits = tl.minimum(tl.maximum(logits, -logit_cap), logit_cap)
    if logits_max_ptr is not None and logits_min_ptr is not None:
        logits_max = tl.load(logits_max_ptr + chunk_start)
        logits_min = tl.load(logits_min_ptr + chunk_start)
        logits = (logits - logits_min) / (logits_max - logits_min)
    label_value = tl.load(labels_ptr + row_idx)
    label_ignore_mask = label_value != label_ignore_value
    logits = tl.where(label_ignore_mask, logits, float("-inf"))
    loss = -tl.log(tl.sum(tl.exp(logits))) / chunk_size
    tl.store(loss_ptr + row_idx, loss)

@triton.jit
def _cross_entropy_backward(
    input_row,
    logit_scaling,
    dlosses_row,
    logits_ptr,
    logits_row_stride,
    labels_ptr,
    label_ignore_value,
    output_row,
    dlosses_ptr,
    logit_cap: tl.constexpr = 20.0,
    logit_shift: tl.constexpr = 0.0,
    logit_scale: tl.constexpr = 1.0,
    fused_size: tl.constexpr = 0,
    logits_max_ptr: tl.constexpr = None,
    logits_min_ptr: tl.constexpr = None,
    row_idx: tl.constexpr = 0,
    BLOCK_SIZE: tl.constexpr = 1,
):
    # Triton kernel to compute gradients with respect to logits
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptr = input_row + col_offsets
    logits_row_offsets = row_idx * logits_row_stride + col_offsets
    logits_mask = col_offsets < fused_size
    logits_row_ptr = logits_ptr + logits_row_offsets
    logits = tl.load(logits_row_ptr, mask=logits_mask, other=float("-inf")).to(
        tl.float32
    )
    if logit_scaling:
        logits = (logits * logit_scale) + logit_shift
    if logit_cap is not None:
        logits = tl.minimum(tl.maximum(logits, -logit_cap), logit_cap)
    if logits_max_ptr is not None and logits_min_ptr is not None:
        logits_max = tl.load(logits_max_ptr)
        logits_min = tl.load(logits_min_ptr)
        logits = (logits - logits_min) / (logits_max - logits_min)
    label_value = tl.load(labels_ptr + row_idx)
    label_ignore_mask = label_value != label_ignore_value
    dlosses = tl.load(dlosses_ptr + row_idx)
