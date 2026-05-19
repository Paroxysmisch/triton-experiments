import triton
import triton.language as tl

# Define the kernel
@triton.jit
def _bgmv_shrink_kernel(
    input_ptr, lora_ptr, out_ptr, lora_indices_ptr,
    scaling, batch_size, N, K,
    stride_input_n, stride_input_k,
    stride_lora_n, stride_lora_k,
    stride_out_n,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_id = pid // (N // BLOCK_N)
    n_block_id = pid % (N // BLOCK_N)

    # Check if the batch should be processed
    lora_index = tl.load(lora_indices_ptr + batch_id)
    if lora_index == -1:
        return

    # Initialize accumulator
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Iterate over K dimension in blocks
    for k_block_id in range(0, K, BLOCK_K * SPLIT_K):
        # Load input and LORA matrices
        input_offset = batch_id * stride_input_k + k_block_id
        lora_offset = lora_index * stride_lora_k + k_block_id

        input_block = tl.load(input_ptr + input_offset)
        lora_block = tl.load(lora_ptr + lora_offset)

        # Element-wise multiplication and reduction
        acc += tl.dot(input_block, lora_block)

    # Scale the result
    acc *= scaling

    # Store the result with atomic addition if necessary
    out_offset = batch_id * stride_out_n + n_block_id * BLOCK_N
    if SPLIT_K > 1:
        tl.atomic_add(out_ptr + out_offset, acc)
    else:
        tl.store(out_ptr + out_offset, acc)

# Define the wrapper function
def _bgmv_shrink(input_tensor, lora_tensor, output_tensor, lora_indices, scaling):
    batch_size, N, K = lora_tensor.shape
    BLOCK_N = 1 << (N - 1).bit_length()  # Next power of two greater than or equal to N
    BLOCK_K = 128  # You can adjust this based on your GPU architecture
    SPLIT_K = 1    # Adjust based on your needs

    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    lora_tensor = lora_tensor.contiguous()
    output_tensor = output_tensor.contiguous()

    # Define strides
    stride_input_n, stride_input_k = input_tensor.stride()
    stride_lora_n, stride_lora_k = lora_tensor.stride()
    stride_out_n = output_tensor.stride(0)

    # Launch the kernel
    grid = (batch_size * (N // BLOCK_N),)
    _bgmv_shrink_kernel[grid](
        input_tensor, lora_tensor, output_tensor, lora_indices,
        scaling, batch_size, N, K,
        stride_input_n, stride_input_k,
        stride_lora_n, stride_lora_k,
        stride_out_n,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_K=SPLIT_K
    )
