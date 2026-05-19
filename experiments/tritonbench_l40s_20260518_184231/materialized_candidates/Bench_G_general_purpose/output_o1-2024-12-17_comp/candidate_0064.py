import torch
import triton
import triton.language as tl


@triton.jit
def _bgmv_expand_slice_kernel(
    input_ptr,       # input matrix
    lora_ptr,        # LoRA weight matrix
    out_ptr,         # output matrix
    lora_indices_ptr,# lora_indices containing batch and index info
    B,               # total batch
    N,               # hidden size
    K,               # dimension per LoRA block
    BLOCK_N: tl.constexpr,   # block size across N
    BLOCK_K: tl.constexpr,   # block size across K
    SPLIT_N: tl.constexpr,   # how many splits in N dimension
    EVEN_K: tl.constexpr,    # flag for even K
    ADD_INPUTS: tl.constexpr,# flag for adding inputs to result
    CAST_TYPE: tl.constexpr  # type casting flag
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    # Range for n in this sub-block
    n_range = tl.arange(0, BLOCK_N)
    # Partial mask to handle boundary
    mask_n = n_start + n_range < N

    # Read the index for current batch from lora_indices
    batch_idx = pid_b
    lora_idx = tl.load(lora_indices_ptr + pid_b)
    # Each pid_b processes one row or segment in the matrix

    # Offsets for input
    # input layout = [B, K], LoRA layout = [N, K], output layout = [B, N]
    # Adjust as needed for correct indexing
    input_offset = batch_idx * K
    lora_offset = lora_idx * K
    out_offset = batch_idx * N

    # Load partial input and LoRA data
    # If EVEN_K is set, we can read more efficiently without boundary checks
    k_range = tl.arange(0, BLOCK_K)
    if EVEN_K:
        input_val = tl.load(
            input_ptr + input_offset + k_range,
            mask=k_range < K
        )
        lora_val = tl.load(
            lora_ptr + lora_offset + n_start * K + k_range[:, None] * N + n_range[None, :],
            mask=(k_range[:, None] < K) & (n_start + n_range[None, :] < N)
        )
    else:
        # Fallback read with explicit boundary checks
        input_val = tl.load(
            input_ptr + input_offset + k_range,
            mask=k_range < K
        )
        lora_val = tl.load(
            lora_ptr + lora_offset + (n_start + n_range[None, :]) * K + k_range[:, None],
            mask=(k_range[:, None] < K) & (n_start + n_range[None, :] < N)
        )

    # Compute partial products: we have shape [BLOCK_K] for input_val
    # and [BLOCK_K, BLOCK_N] for lora_val
    # Expand input_val along axis=1 to match lora_val broadcast
    input_val_broadcast = tl.broadcast_to(input_val.reshape((BLOCK_K, 1)), (BLOCK_K, BLOCK_N))
    partial_product = input_val_broadcast * lora_val

    # Sum across K dimension
    accum = tl.sum(partial_product, axis=0)

    # Optionally cast accum
    if CAST_TYPE:
        accum = accum.to(tl.float32)

    # Optionally add the existing input_val to output
    # This is a per-element add if ADD_INPUTS is set
    if ADD_INPUTS:
        old = tl.load(
            out_ptr + out_offset + n_start + n_range,
            mask=mask_n
        )
        accum = accum + old

    # Store result
    tl.store(
        out_ptr + out_offset + n_start + n_range,
        accum,
        mask=mask_n
    )


@torch.inference_mode()
def _bgmv_expand_slice(
    input_tensor: torch.Tensor,
    lora_tensor: torch.Tensor,
    out_tensor: torch.Tensor,
    lora_indices: torch.Tensor,
    BLOCK_N=64,
    BLOCK_K=32,
    SPLIT_N=1,
    EVEN_K=True,
    ADD_INPUTS=False,
    CAST_TYPE=False
):
    assert input_tensor.is_cuda and lora_tensor.is_cuda and out_tensor.is_cuda and lora_indices.is_cuda, \
        "All tensors must be on CUDA"
    B = lora_indices.shape[0]
    assert input_tensor.ndim == 2 and lora_tensor.ndim == 2, \
        "input_tensor must be [B, K], lora_tensor must be [N, K]"
    K = input_tensor.shape[1]
    N = lora_tensor.shape[0]
    assert out_tensor.shape == (B, N), \
        "out_tensor shape must match (B, N)"

    grid = ( (N + BLOCK_N - 1) // BLOCK_N, B )
    _bgmv_expand_slice_kernel[grid](
        input_tensor, lora_tensor, out_tensor, lora_indices,
        B, N, K,
        BLOCK_N, BLOCK_K, SPLIT_N, EVEN_K, ADD_INPUTS, CAST_TYPE
    )
