import torch
import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_slice_kernel(
    # Pointers to matrices
    input_ptr, lora_ptr, out_ptr,
    # Matrix dimensions
    batch, hidden_size, r,
    # Strides for the different matrices
    input_batch_stride, input_row_stride,
    lora_batch_stride, lora_row_stride,
    output_batch_stride, output_row_stride,
    # Additional parameters
    lora_indices_ptr,
    # Constants
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_N: tl.constexpr, EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr, CAST_TYPE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute batch index and N index
    batch_id = pid // SPLIT_N
    n_id = pid % SPLIT_N

    # Compute the start offset for this block
    n_start = n_id * BLOCK_N
    k_start = 0

    # Load the lora index for this batch
    lora_idx = tl.load(lora_indices_ptr + batch_id)
    
    # Initialize pointers for this batch
    input_batch_ptr = input_ptr + batch_id * input_batch_stride
    lora_batch_ptr = lora_ptr + lora_idx * lora_batch_stride
    output_batch_ptr = out_ptr + batch_id * output_batch_stride

    # Initialize accumulator
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Make block pointers
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Mask for bounds checking
    n_mask = offs_n < hidden_size

    # Loop over k dimension
    for k in range(0, r, BLOCK_K):
        # Load inputs
        x = tl.load(input_batch_ptr + offs_k[None, :] * input_row_stride,
                   mask=offs_k[None, :] < r, other=0.0)
        
        # Load weights
        w = tl.load(lora_batch_ptr + offs_n[:, None] * lora_row_stride + offs_k[None, :],
                   mask=(offs_k[None, :] < r) & (offs_n[:, None] < hidden_size),
                   other=0.0)

        # Compute matrix multiplication
        acc += tl.sum(w * x[None, :], axis=1)

    # Write output
    if ADD_INPUTS:
        if CAST_TYPE:
            # Load and add existing output
            existing = tl.load(output_batch_ptr + offs_n * output_row_stride,
                             mask=n_mask, other=0.0)
            output = existing + acc
        else:
            output = acc
    else:
        output = acc

    # Store the result
    tl.store(output_batch_ptr + offs_n * output_row_stride,
             output, mask=n_mask)

@torch.inference_mode()
def _bgmv_expand_slice(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    lora_indices: torch.Tensor,
    add_inputs: bool = True,
    cast_type: bool = True,
):
    batch, r = input.shape
    _, hidden_size = weight.shape
    
    # Assert shapes
    assert input.shape == (batch, r)
    assert weight.shape == (r, hidden_size)
    assert output.shape == (batch, hidden_size)
    assert lora_indices.shape == (batch,)
    
    # Define block sizes
    BLOCK_N = 128
    BLOCK_K = 32
    
    # Calculate grid size
    split_n = triton.cdiv(hidden_size, BLOCK_N)
    grid = (batch * split_n,)
    
    # Launch kernel
    _bgmv_expand_slice_kernel[grid](
        input, weight, output,
        batch, hidden_size, r,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        lora_indices,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        SPLIT_N=split_n, EVEN_K=(r % BLOCK_K == 0),
        ADD_INPUTS=add_inputs, CAST_TYPE=cast_type,
    )

    return output
