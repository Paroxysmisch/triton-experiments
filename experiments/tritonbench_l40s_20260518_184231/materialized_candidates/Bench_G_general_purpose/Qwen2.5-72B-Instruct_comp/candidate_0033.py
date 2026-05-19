import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K, Out, Out_scale, DestLoc,
    stride_k_batch, stride_k_head, stride_k_dmodel,
    stride_out_batch, stride_out_head, stride_out_dmodel,
    stride_out_scale_batch, stride_out_scale_head,
    head_num, seq_len, block_dmodel: tl.constexpr, block_head: tl.constexpr
):
    pid = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    batch_id = tl.program_id(axis=2)

    # Calculate the base pointers for K and Out
    k_base = K + batch_id * stride_k_batch + head_id * stride_k_head
    out_base = Out + batch_id * stride_out_batch + head_id * stride_out_head
    out_scale_base = Out_scale + batch_id * stride_out_scale_batch + head_id * stride_out_scale_head

    # Initialize the maximum value for the head
    max_val = tl.zeros([1], dtype=tl.float32)

    # Load the data for the current head
    for i in range(0, seq_len, block_dmodel):
        k_ptr = k_base + i * stride_k_dmodel
        k_block = tl.load(k_ptr, mask=i + tl.arange(0, block_dmodel) < seq_len, other=0.0)
        max_val = tl.maximum(max_val, tl.max(tl.abs(k_block), axis=0))

    # Calculate the scaling factor
    scale = 127.0 / max_val
    tl.store(out_scale_base, scale)

    # Quantize and store the data
    for i in range(0, seq_len, block_dmodel):
        k_ptr = k_base + i * stride_k_dmodel
        k_block = tl.load(k_ptr, mask=i + tl.arange(0, block_dmodel) < seq_len, other=0.0)
        quantized_block = tl.cast(tl.round(k_block * scale), tl.int8)
        dest_index = tl.load(DestLoc + i, mask=i + tl.arange(0, block_dmodel) < seq_len, other=0)
        out_ptr = out_base + dest_index * stride_out_dmodel
        tl.store(out_ptr, quantized_block, mask=i + tl.arange(0, block_dmodel) < seq_len)

import torch

def destindex_copy_quantize_kv(K, Out, Out_scale, DestLoc, head_num, seq_len, block_dmodel, block_head):
    # Get the dimensions of the input tensors
    batch_size, _, dmodel = K.shape

    # Define the grid and block sizes
    grid = (seq_len // block_dmodel, head_num, batch_size)
    block = (block_dmodel, block_head, 1)

    # Define the strides for the input and output tensors
    stride_k_batch = K.stride(0)
    stride_k_head = K.stride(1)
    stride_k_dmodel = K.stride(2)

    stride_out_batch = Out.stride(0)
    stride_out_head = Out.stride(1)
    stride_out_dmodel = Out.stride(2)

    stride_out_scale_batch = Out_scale.stride(0)
    stride_out_scale_head = Out_scale.stride(1)

    # Launch the kernel
    _fwd_kernel_destindex_copy_quantize_kv[grid, block](
        K, Out, Out_scale, DestLoc,
        stride_k_batch, stride_k_head, stride_k_dmodel,
        stride_out_batch, stride_out_head, stride_out_dmodel,
        stride_out_scale_batch, stride_out_scale_head,
        head_num, seq_len, block_dmodel, block_head
    )
