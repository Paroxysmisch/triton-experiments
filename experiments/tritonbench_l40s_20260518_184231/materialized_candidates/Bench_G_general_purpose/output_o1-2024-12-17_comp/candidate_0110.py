import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_kernel(
    input_ptr,  # *F32 or BF16
    lora_ptr,   # *F32 or BF16
    out_ptr,    # *F32
    lora_indices_ptr,  # *int32
    N, K,
    stride_in_batch,
    stride_in_n,
    stride_in_k,
    stride_lora_batch,
    stride_lora_n,
    stride_lora_k,
    stride_out_batch,
    stride_out_n,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    ADD_INPUT: tl.constexpr
):
    batch_id = tl.program_id(0)
    split_id = tl.program_id(1)
    n_offset = split_id * BLOCK_N

    # Each program loads a single batch row
    lora_index = tl.load(lora_indices_ptr + batch_id)
    # if lora_index == -1, skip
    skip_mask = lora_index == -1

    # Create pointer offsets
    # input and lora are [Batch, K], output is [Batch, N], lora is also [N, K] or [K, N] depending on usage
    # Some LoRA setups store [r, N], but here we assume [K, N] is the relevant shape
    # We'll do a gemv-like load across 'k' dimension
    # load K in chunks
    k_offsets = tl.arange(0, BLOCK_K)
    n_offsets = tl.arange(0, BLOCK_N) + n_offset

    # partial sum
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # We iterate over K by BLOCK_K
    # We read from input (which is a single vector of length K, offset by batch)
    # We read from lora (which is an NxK or KxN matrix depending on usage)
    # We'll assume a typical row=K, col=N scenario for performance
    # We'll proceed for the gemv: out[batch_id, n] += sum_{k} (input[batch_id, k] * lora[k, n])
    for k_block_start in range(0, K, BLOCK_K):
        # If skip, break with a zero update
        if skip_mask:
            break
        # load input
        inp_ptrs = input_ptr + batch_id * stride_in_batch \
                              + (k_block_start + k_offsets) * stride_in_k
        if CAST_TYPE:
            inp_vals = tl.load(inp_ptrs, mask=k_block_start + k_offsets < K, other=0.0).to(tl.float32)
        else:
            inp_vals = tl.load(inp_ptrs, mask=k_block_start + k_offsets < K, other=0.0)

        # load lora
        # the index might shift which row we load from
        lora_batch_offset = lora_index * stride_lora_batch
        lora_k_ptrs = lora_ptr + lora_batch_offset \
                              + (k_block_start + k_offsets) * stride_lora_k
        lora_vals = tl.load(
            lora_k_ptrs[:, None] + (n_offsets[None, :] * stride_lora_n),
            mask=(k_block_start + k_offsets[:, None] < K) & (n_offsets[None, :] < N),
            other=0.0
        )
        # cast to float32 if needed
        if CAST_TYPE:
            lora_vals = lora_vals.to(tl.float32)

        # multiply accumulate
        # lora_vals has shape [BLOCK_K, BLOCK_N], inp_vals has shape [BLOCK_K]
        # broadcast inp_vals along axis 1
        acc += tl.sum(lora_vals * inp_vals[:, None], axis=0)

    # store to out
    out_ptrs = out_ptr + batch_id * stride_out_batch + n_offsets * stride_out_n
    if skip_mask:
        if ADD_INPUT:
            # if skipping, just load input and store for final
            # user might want out += input or out = input
            # but if skipping, add_input means we do out[n] += input[batch_id, n]
            # we get input from input_ptr shape [Batch, n] dimension doesn't exist, so we skip
            # if needed, do out = 0 or out = out
            pass
        # no update if skipping
    else:
        tmp = acc
        if ADD_INPUT:
            # we add user input to the result
            # user input shape is [Batch, N] - let's load and add
            add_in_ptrs = out_ptrs     # reuse out memory location
            orig_out = tl.load(add_in_ptrs, mask=n_offsets < N, other=0.0)
            tmp += orig_out
        tl.store(out_ptrs, tmp, mask=n_offsets < N)

@triton.jit
def _copy_input_to_out_if_skipped(
    input_ptr, out_ptr,
    batch_idx, length,
    stride_in_batch, stride_in_n,
    stride_out_batch, stride_out_n,
    BLOCK_SIZE: tl.constexpr
):
    # optional: if skipping LoRA broadcast, copy input to out or do nothing
    # placeholder if needed, else can skip
    pass

def _bgmv_expand(
    input_tensor,
    lora_tensor,
    out_tensor,
    lora_indices,
    split_n=1,
    add_input=True
):
    """
    A wrapper to launch _bgmv_expand_kernel.
    input_tensor:  [batch, K]
    lora_tensor:   [num_lora, K, N] or [num_lora, K, N] in row-major
    out_tensor:    [batch, N]
    lora_indices:  [batch] int
    split_n: how many splits on N dimension
    add_input: conditionally add input to out
    """
    import math
    # Ensure contiguity and get shapes
    input_ = input_tensor.contiguous()
    lora_ = lora_tensor.contiguous()
    out_ = out_tensor.contiguous()
    indices_ = lora_indices.contiguous()

    batch_size, K = input_.shape
    # assume lora_.shape is [num_lora, K, N]
    num_lora, K_lora, N = lora_.shape
    assert K == K_lora
    BLOCK_K = 128
    BLOCK_N = 128
    # decide if cast is needed
    CAST_TYPE = (input_.dtype in (triton.language.bfloat16,)) or \
                (lora_.dtype in (triton.language.bfloat16,))

    # convert everything to pointer
    input_ptr = input_.data_ptr()
    lora_ptr = lora_.data_ptr()
    out_ptr = out_.data_ptr()
    lora_indices_ptr = indices_.data_ptr()

    # Strides
    stride_in_batch = input_.stride(0)
    stride_in_n = 0
    stride_in_k = input_.stride(1)

    stride_lora_batch = lora_.stride(0)
    stride_lora_k = lora_.stride(1)
    stride_lora_n = lora_.stride(2)

    stride_out_batch = out_.stride(0)
    stride_out_n = out_.stride(1)

    grid = (batch_size, split_n)
    _bgmv_expand_kernel[grid](
        input_ptr,
        lora_ptr,
        out_ptr,
        lora_indices_ptr,
        N, K,
        stride_in_batch,
        stride_in_n,
        stride_in_k,
        stride_lora_batch,
        stride_lora_n,
        stride_lora_k,
        stride_out_batch,
        stride_out_n,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        CAST_TYPE=CAST_TYPE,
        ADD_INPUT=add_input
    )
    return out_
