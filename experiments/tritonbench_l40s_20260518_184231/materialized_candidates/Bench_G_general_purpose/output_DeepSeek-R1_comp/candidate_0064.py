import torch
import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_slice_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    lora_indices_ptr,
    batch_size,
    in_features,
    out_features,
    stride_input_batch,
    stride_input_feature,
    stride_lora_lora,
    stride_lora_out,
    stride_lora_in,
    stride_out_batch,
    stride_out_feature,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_split_n = tl.program_id(1)

    # Check if batch is within range
    if pid_batch >= batch_size:
        return

    # Get LoRA index for current batch
    lora_idx = tl.load(lora_indices_ptr + pid_batch)

    # Calculate the start offset for the current N block
    n_start = pid_split_n * BLOCK_N
    n_mask = n_start + tl.arange(0, BLOCK_N)
    valid_n = n_mask < out_features

    # Initialize accumulator
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K dimension in blocks of BLOCK_K
    k_start = 0
    num_k_blocks = tl.cdiv(in_features, BLOCK_K)
    for _ in range(num_k_blocks):
        k_offset = k_start + tl.arange(0, BLOCK_K)
        valid_k = k_offset < in_features if not EVEN_K else True

        # Load input values [BLOCK_K]
        input_ptr_batch = input_ptr + pid_batch * stride_input_batch
        input_val = tl.load(
            input_ptr_batch + k_offset,
            mask=valid_k,
            other=0.0,
        )

        # Cast input if required
        if CAST_TYPE:
            input_val = input_val.to(tl.float16)

        # Load LoRA weights [BLOCK_N, BLOCK_K]
        lora_ptr_lora = lora_ptr + lora_idx * stride_lora_lora
        lora_ptr_kn = lora_ptr_lora + (n_mask[:, None] * stride_lora_out + k_offset[None, :] * stride_lora_in)
        lora_val = tl.load(
            lora_ptr_kn,
            mask=valid_n[:, None] & (valid_k if not EVEN_K else True),
            other=0.0,
        )

        # Cast LoRA weights if required
        if CAST_TYPE:
            lora_val = lora_val.to(tl.float16)

        # Compute partial sum
        partial = tl.sum(input_val[None, :] * lora_val, axis=1)
        acc += partial.to(tl.float32)

        k_start += BLOCK_K

    # Write back to output
    out_ptr_batch = out_ptr + pid_batch * stride_out_batch
    out_offset = n_start + tl.arange(0, BLOCK_N)

    if ADD_INPUTS:
        current_val = tl.load(
            out_ptr_batch + out_offset,
            mask=valid_n,
            other=0.0,
        ).to(tl.float32)
        acc += current_val

    tl.store(
        out_ptr_batch + out_offset,
        acc.to(input_val.dtype if CAST_TYPE else tl.float32),
        mask=valid_n,
    )

@torch.inference_mode()
def _bgmv_expand_slice(
    input: torch.Tensor,
    lora: torch.Tensor,
    out: torch.Tensor,
    lora_indices: torch.Tensor,
    BLOCK_N: int,
    BLOCK_K: int,
    EVEN_K: bool,
    ADD_INPUTS: bool,
    CAST_TYPE: bool,
):
    # Validate tensor dimensions
    assert input.dim() == 2, "Input must be 2D"
    assert lora.dim() == 3, "LoRA must be 3D"
    assert out.dim() == 2, "Output must be 2D"
    assert lora_indices.dim() == 1, "lora_indices must be 1D"

    batch_size, in_features = input.shape
    out_batch, out_features = out.shape
    num_loras, lora_out, lora_in = lora.shape

    assert batch_size == out_batch, "Mismatched batch size"
    assert lora_out == out_features, "Mismatched LoRA output features"
    assert lora_in == in_features, "Mismatched LoRA input features"
    assert (lora_indices < num_loras).all(), "Invalid LoRA indices"

    # Ensure contiguous tensors
    input = input.contiguous()
    lora = lora.contiguous()
    out = out.contiguous()
    lora_indices = lora_indices.contiguous().to(torch.int32)

    # Compute grid dimensions
    grid_n = (out_features + BLOCK_N - 1) // BLOCK_N
    grid = (batch_size, grid_n)

    # Launch kernel
    _bgmv_expand_slice_kernel[grid](
        input,
        lora,
        out,
        lora_indices,
        batch_size,
        in_features,
        out_features,
        input.stride(0),
        input.stride(1),
        lora.stride(0),
        lora.stride(1),
        lora.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        EVEN_K=EVEN_K,
        ADD_INPUTS=ADD_INPUTS,
        CAST_TYPE=CAST_TYPE,
    )

    return out
