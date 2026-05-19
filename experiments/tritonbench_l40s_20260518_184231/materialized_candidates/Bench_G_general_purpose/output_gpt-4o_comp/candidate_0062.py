import triton
import triton.language as tl
import math

# Triton kernel for embedding lookup
@triton.jit
def embedding_kernel(
    token_ids_ptr,  # Pointer to token IDs tensor (sequence of token IDs)
    weight_ptr,     # Pointer to weight matrix (embedding matrix)
    out_ptr,        # Pointer to output tensor (embedding vectors)
    stride_w_d,     # Stride for weight matrix in dimension D (model dimension)
    stride_o_d,     # Stride for output tensor in dimension D (model dimension)
    seq_len,        # Length of the sequence (number of tokens in the sequence)
    d_model,        # Dimension of the model (embedding vector size)
    BLOCK_N: tl.constexpr,  # Block size for sequence processing
    BLOCK_DMODEL: tl.constexpr  # Block size for model dimension
):
    # Program IDs
    pid_n = tl.program_id(0)  # Sequence index
    pid_d = tl.program_id(1)  # Model dimension block index

    # Offsets for sequence processing
    seq_start = pid_n * BLOCK_N
    seq_offsets = seq_start + tl.arange(0, BLOCK_N)

    # Offsets for model dimension processing
    d_start = pid_d * BLOCK_DMODEL
    d_offsets = d_start + tl.arange(0, BLOCK_DMODEL)

    # Mask to ensure valid token indices
    seq_mask = seq_offsets < seq_len
    seq_offsets = tl.where(seq_mask, seq_offsets, 0)

    # Load token IDs
    token_ids = tl.load(token_ids_ptr + seq_offsets, mask=seq_mask, other=0)

    # Compute offsets in the weight matrix
    weight_offsets = token_ids[:, None] * stride_w_d + d_offsets[None, :]
    weight_mask = (seq_mask[:, None] & (d_offsets[None, :] < d_model))

    # Load embeddings from the weight matrix
    embeddings = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)

    # Compute offsets in the output tensor
    out_offsets = seq_offsets[:, None] * stride_o_d + d_offsets[None, :]
    out_mask = (seq_mask[:, None] & (d_offsets[None, :] < d_model))

    # Store embeddings into the output tensor
    tl.store(out_ptr + out_offsets, embeddings, mask=out_mask)


# Wrapper function for configuring and invoking the kernel
def embedding(token_ids, weight, BLOCK_N=128, BLOCK_DMODEL=128):
    """
    Extracts embeddings for a sequence of token IDs from the weight matrix.

    Args:
        token_ids (torch.Tensor): Input tensor of token IDs (shape: [seq_len]).
        weight (torch.Tensor): Weight matrix (shape: [vocab_size, d_model]).
        BLOCK_N (int): Block size for sequence processing.
        BLOCK_DMODEL (int): Block size for model dimension processing.
    
    Returns:
        torch.Tensor: Output tensor of embeddings (shape: [seq_len, d_model]).
    """
    import torch

    # Ensure dimensions are compatible
    seq_len = token_ids.shape[0]
    vocab_size, d_model = weight.shape

    # Align BLOCK_DMODEL to the next power of two
    BLOCK_DMODEL = 2 ** math.ceil(math.log2(BLOCK_DMODEL))

    # Allocate output tensor
    out = torch.empty((seq_len, d_model), device=token_ids.device, dtype=weight.dtype)

    # Strides
    stride_w_d = weight.stride(1)  # Stride for weight matrix in dimension D
    stride_o_d = out.stride(1)     # Stride for output tensor in dimension D

    # Configure grid size
    grid = (math.ceil(seq_len / BLOCK_N), math.ceil(d_model / BLOCK_DMODEL))

    # Launch the Triton kernel
    embedding_kernel[grid](
        token_ids_ptr=token_ids.data_ptr(),
        weight_ptr=weight.data_ptr(),
        out_ptr=out.data_ptr(),
        stride_w_d=stride_w_d,
        stride_o_d=stride_o_d,
        seq_len=seq_len,
        d_model=d_model,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL
    )

    return out
