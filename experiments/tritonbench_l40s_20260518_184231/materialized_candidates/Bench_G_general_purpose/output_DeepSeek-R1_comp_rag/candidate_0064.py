import triton
import triton.language as tl
import torch

@triton.jit
def embedding_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_batch_stride,
    input_seq_stride,
    weight_stride_0,
    weight_stride_1,
    output_batch_stride,
    output_seq_stride,
    output_dmodel_stride,
    seq_len,
    d_model,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CONTIGUOUS_INPUT: tl.constexpr,
    IS_CONTIGUOUS_WEIGHT: tl.constexpr,
    IS_CONTIGUOUS_OUTPUT: tl.constexpr,
):
    # Extract 3D grid coordinates
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    pid_dmodel = tl.program_id(2)

    # Create ranges for sequence and embedding dimension blocks
    seq_offsets = pid_seq * BLOCK_N + tl.arange(0, BLOCK_N)
    dmodel_offsets = pid_dmodel * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)

    # Create masks for valid sequences and embedding dimensions
    mask_seq = seq_offsets < seq_len
    mask_dmodel = dmodel_offsets < d_model

    # Calculate input offsets and load token IDs
    input_offsets = pid_batch * input_batch_stride + seq_offsets * input_seq_stride
    token_ids = tl.load(input_ptr + input_offsets, mask=mask_seq, other=0)

    # Calculate weight offsets using broadcasting
    weight_row_offsets = token_ids[:, None] * weight_stride_0
    weight_col_offsets = dmodel_offsets[None, :] * weight_stride_1
    weight_offsets = weight_row_offsets + weight_col_offsets

    # Load embeddings with combined mask
    combined_mask = mask_seq[:, None] & mask_dmodel[None, :]
    embeddings = tl.load(weight_ptr + weight_offsets, mask=combined_mask, other=0.0)

    # Calculate output offsets using tensor strides
    batch_offset = pid_batch * output_batch_stride
    seq_offset = seq_offsets[:, None] * output_seq_stride
    dmodel_offset = dmodel_offsets[None, :] * output_dmodel_stride
    output_offsets = batch_offset + seq_offset + dmodel_offset

    # Store results with proper masking
    tl.store(output_ptr + output_offsets, embeddings, mask=combined_mask)

def embedding(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    BLOCK_N: int = 128
) -> torch.Tensor:
    # Validate input dimensions
    assert input.dim() == 2, "Input must be 2D (batch, seq_len)"
    assert weight.dim() == 2, "Weight must be 2D (vocab_size, d_model)"
    
    batch_size, seq_len = input.shape
    vocab_size, d_model = weight.shape

    # Initialize output tensor if not provided
    if out is None:
        out = torch.empty((batch_size, seq_len, d_model), 
                         device=input.device, dtype=weight.dtype)
    else:
        assert out.shape == (batch_size, seq_len, d_model), "Invalid output shape"

    # Determine optimal block size for embedding dimension
    BLOCK_DMODEL = triton.next_power_of_2(d_model)
    
    # Configure kernel launch parameters
    grid = (
        batch_size,  # Parallel over batches
        triton.cdiv(seq_len, BLOCK_N),  # Parallel over sequence blocks
        triton.cdiv(d_model, BLOCK_DMODEL),  # Parallel over embedding dim
    )

    # Get tensor stride information
    strides = {
        'input_batch_stride': input.stride(0),
        'input_seq_stride': input.stride(1),
        'weight_stride_0': weight.stride(0),
        'weight_stride_1': weight.stride(1),
        'output_batch_stride': out.stride(0),
        'output_seq_stride': out.stride(1),
        'output_dmodel_stride': out.stride(2),
    }

    # Launch kernel with optimized configuration
    embedding_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        output_ptr=out,
        seq_len=seq_len,
        d_model=d_model,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CONTIGUOUS_INPUT=input.is_contiguous(),
        IS_CONTIGUOUS_WEIGHT=weight.is_contiguous(),
        IS_CONTIGUOUS_OUTPUT=out.is_contiguous(),
        **strides
    )
    
    return out
