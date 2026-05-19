import torch
import triton
import triton.language as tl
from deepspeed.accelerator import get_accelerator

@triton.jit
def _sgmv_expand_slice_kernel(
    input_ptr,
    lora_ptr,
    output_ptr,
    seq_lengths_ptr,
    lora_indices_ptr,
    batch_size,
    max_seq_length,
    hidden_size,
    inter_size,
    K,
    stride_input_batch,
    stride_input_seq,
    stride_input_hidden,
    stride_output_batch,
    stride_output_seq,
    stride_output_hidden,
    stride_seq_lengths_batch,
    stride_seq_lengths_seq,
    stride_lora_indices_batch,
    stride_lora_indices_lo,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CAST_D: tl.constexpr,
    HAS_SEQ_LENGTHS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Implements the specialized form of matrix multiplication
    for sparse Generalized Matrix-Vector Multiplication (SGMV).
    """
    pid_b = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    input_offset = pid_b * stride_input_batch + pid_n * BLOCK_N * stride_input_hidden
    output_offset = pid_b * stride_output_batch + pid_n * BLOCK_N * stride_output_hidden
    lora_index_offset = pid_b * stride_lora_indices_batch
    seq_length_offset = pid_b * stride_seq_lengths_batch

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K) + pid_k * BLOCK_K

    a_ptr = input_ptr + input_offset + offs_m[:, None] * stride_input_seq + offs_k[None, :] * stride_input_hidden
    b_ptr = lora_ptr + offs_k[:, None] * inter_size + offs_n[None, :] * stride_lora_indices_lo
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if HAS_SEQ_LENGTHS:
        seq_lengths = tl.load(seq_lengths_ptr + seq_length_offset + offs_m)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if HAS_SEQ_LENGTHS:
            mask_k = offs_k[None, :] < seq_lengths[None, :]
        else:
            mask_k = True
        if CAST_D:
            a = tl.load(a_ptr, mask=mask_k[None, :], other=0).to(tl.float16)
            b = tl.load(b_ptr, mask=mask_k[:, None], other=0).to(tl.float16)
        else:
            a = tl.load(a_ptr, mask=mask_k[None, :], other=0).to(tl.float32)
            b = tl.load(b_ptr, mask=mask_k[:, None], other=0).to(tl.float32)
        accumulator += tl.dot(a, b)
        a_ptr += BLOCK_K * stride_input_hidden
        b_ptr += BLOCK_K * stride_lora_indices_lo

    lo_hidden_size = tl.load(lora_indices_ptr + lora_index_offset + offs_n)

    offsets = output_ptr + output_offset + lo_hidden_size[:, None] * stride_output_seq + offs_m[None, :] * stride_output_hidden
    mask = offs_n[None, :] < hidden_size
    if CAST_D:
        tl.store(offsets, accumulator.to(tl.float16), mask=mask)
    else:
        tl.store(offsets, accumulator.to(tl.float32), mask=mask)


def _sgmv_expand_slice(
    inputs: torch.Tensor,
    lora_indices: torch.Tensor,
    lora_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    seq_lengths: Optional[torch.Tensor],
    cast_dtype: Optional[torch.dtype] = None,
) -> None:
    """
    Arguments:
        inputs: input tensor
        lora_indices: the LoRA's indices.
        lora_weights: the LoRA's weights.
        output_tensor: output tensor
        seq_lengths: sequence lengths
        cast_dtype: cast to this dtype (optional)
    """
    assert inputs.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert lora_weights.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert inputs.size(1) == lora_indices.size(0)
    assert inputs.size(2) == lora_indices.size(1)
    assert lora_weights.size(0) == lora_indices.size(0)
    assert inputs.is_contiguous()
    assert lora_indices.is_contiguous()
    assert lora_weights.is_contiguous()
    assert output_tensor.is_contiguous()

    if cast_dtype is None:
        cast_dtype = inputs.dtype

    if lora_weights.dtype != cast_dtype:
        lora_weights = lora_weights.to(cast_dtype)

    batch_size, seq_len, hidden_size = inputs.shape
    inter_size = lora_weights.shape[1]
    K = inputs.shape[2]
    max_seq_length = seq_len if seq_lengths is None else torch.max(seq_lengths).item()

    grid = lambda META: (
        triton.cdiv(hidden_size, META["BLOCK_N"]),
        triton.cdiv(K, META["BLOCK_K"]),
        triton.cdiv(hidden_size, META["BLOCK_N"]),
    )

    kernel_meta = {
        "BLOCK_M": 256,
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "CAST_D": lora_weights.dtype == torch.float16,
        "HAS_SEQ_LENGTHS": seq_lengths is not None,
        "ALLOW_TF32": triton.is_tf32_enabled(),
    }

    kernel_meta["BLOCK_N"] = min(kernel_meta["BLOCK_N"], hidden_size)
    kernel_meta["BLOCK_M"] = min(kernel_meta["BLOCK_M"], get_accelerator().warp_size(0) * 256)

    _sgmv_expand_slice_kernel[grid](
        inputs,
        lora_weights,
        output_tensor,
        seq_lengths,
        lora_indices,
        batch_size,
        max_seq_length,
        hidden_size,
        inter_size,
        K,
        inputs.str
