import triton
import triton.language as tl

@triton.jit
def _sgmv_expand_slice_kernel(
    x_ptr,  # pointer to the input matrix
    w_ptr,  # pointer to the LoRA weights
    lora_indices_ptr,  # pointer to the LoRA indices
    out_ptr,  # pointer to the output matrix
    x_batch_stride,  # stride for the batch dimension of x
    x_seq_stride,  # stride for the sequence dimension of x
    x_hidden_stride,  # stride for the hidden dimension of x
    w_batch_stride,  # stride for the batch dimension of w
    w_seq_stride,  # stride for the sequence dimension of w
    w_hidden_stride,  # stride for the hidden dimension of w
    out_batch_stride,  # stride for the batch dimension of out
    out_seq_stride,  # stride for the sequence dimension of out
    out_hidden_stride,  # stride for the hidden dimension of out
    batch_size,  # batch size
    max_seq_len,  # maximum sequence length
    hidden_dim,  # hidden dimension
    lora_rank,  # LoRA rank
    BLOCK_M: tl.constexpr,  # block size for the sequence dimension
    BLOCK_N: tl.constexpr,  # block size for the hidden dimension
    BLOCK_K: tl.constexpr,  # block size for the LoRA rank
):
    # Compute the batch and sequence indices
    pid = tl.program_id(axis=0)
    batch_id = pid // (max_seq_len // BLOCK_M)
    seq_id = (pid % (max_seq_len // BLOCK_M)) * BLOCK_M

    # Compute the output block pointer
    out_block_ptr = out_ptr + batch_id * out_batch_stride + seq_id * out_seq_stride

    # Zero initialize the output block
    out_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over the LoRA rank
    for k in range(0, lora_rank, BLOCK_K):
        # Load the input block
        x_block_ptr = x_ptr + batch_id * x_batch_stride + seq_id * x_hidden_stride
        x_block = tl.load(x_block_ptr, mask=seq_id + tl.arange(0, BLOCK_M) < max_seq_len, other=0.0)

        # Load the LoRA weight block
        lora_indices = tl.load(lora_indices_ptr + batch_id * lora_rank + k, mask=k + tl.arange(0, BLOCK_K) < lora_rank, other=0)
        w_block_ptr = w_ptr + batch_id * w_batch_stride + lora_indices * w_hidden_stride
        w_block = tl.load(w_block_ptr, mask=k + tl.arange(0, BLOCK_K) < lora_rank, other=0.0)

        # Perform the matrix multiplication
        out_block += tl.dot(x_block, w_block)

    # Store the output block
    tl.store(out_block_ptr, out_block, mask=seq_id + tl.arange(0, BLOCK_M) < max_seq_len)

import torch
import triton
import triton.language as tl

def _sgmv_expand_slice(x, w, lora_indices, out, BLOCK_M, BLOCK_N, BLOCK_K):
    # Validate input shapes and data types
    assert x.dim() == 3, "Input x must be a 3D tensor (batch, seq, hidden)"
    assert w.dim() == 3, "Input w must be a 3D tensor (batch, lora_rank, hidden)"
    assert lora_indices.dim() == 2, "Input lora_indices must be a 2D tensor (batch, lora_rank)"
    assert out.dim() == 3, "Output out must be a 3D tensor (batch, seq, hidden)"
    assert x.dtype == torch.float32, "Input x must be of type float32"
    assert w.dtype == torch.float32, "Input w must be of type float32"
    assert lora_indices.dtype == torch.int32, "Input lora_indices must be of type int32"
    assert out.dtype == torch.float32, "Output out must be of type float32"

    # Ensure tensors are contiguous
    x = x.contiguous()
    w = w.contiguous()
    lora_indices = lora_indices.contiguous()
    out = out.contiguous()

    # Get the dimensions
    batch_size, max_seq_len, hidden_dim = x.shape
    _, lora_rank, _ = w.shape

    # Set up the Triton launch grid
    grid = (batch_size * (max_seq_len // BLOCK_M),)

    # Launch the kernel
    _sgmv_expand_slice_kernel[grid](
        x, w, lora_indices, out,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        batch_size, max_seq_len, hidden_dim, lora_rank,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
