import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K_ptr,  # Pointer to the input tensor K
    Out_ptr,  # Pointer to the output tensor Out
    Out_scale_ptr,  # Pointer to the output tensor Out_scale
    Dest_loc_ptr,  # Pointer to the destination indices tensor Dest_loc
    K_batch_stride,  # Stride for batch dimension in K
    K_head_stride,  # Stride for head dimension in K
    K_seq_stride,  # Stride for sequence dimension in K
    K_hidden_stride,  # Stride for hidden dimension in K
    Out_batch_stride,  # Stride for batch dimension in Out
    Out_head_stride,  # Stride for head dimension in Out
    Out_seq_stride,  # Stride for sequence dimension in Out
    Out_hidden_stride,  # Stride for hidden dimension in Out
    Out_scale_batch_stride,  # Stride for batch dimension in Out_scale
    Out_scale_head_stride,  # Stride for head dimension in Out_scale
    Out_scale_seq_stride,  # Stride for sequence dimension in Out_scale
    Dest_loc_batch_stride,  # Stride for batch dimension in Dest_loc
    Dest_loc_head_stride,  # Stride for head dimension in Dest_loc
    Dest_loc_seq_stride,  # Stride for sequence dimension in Dest_loc
    batch_size,  # Batch size
    num_heads,  # Number of heads
    seq_len,  # Sequence length
    hidden_dim,  # Hidden dimension
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelism
):
    # Compute the batch, head, and sequence indices for the current thread block
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)

    # Check if the indices are within bounds
    if batch_idx >= batch_size or head_idx >= num_heads or seq_idx >= seq_len:
        return

    # Compute the base pointers for the current batch, head, and sequence
    K_base_ptr = K_ptr + batch_idx * K_batch_stride + head_idx * K_head_stride + seq_idx * K_seq_stride
    Out_base_ptr = Out_ptr + batch_idx * Out_batch_stride + head_idx * Out_head_stride
    Out_scale_base_ptr = Out_scale_ptr + batch_idx * Out_scale_batch_stride + head_idx * Out_scale_head_stride
    Dest_loc_base_ptr = Dest_loc_ptr + batch_idx * Dest_loc_batch_stride + head_idx * Dest_loc_head_stride + seq_idx * Dest_loc_seq_stride

    # Load the destination index
    dest_idx = tl.load(Dest_loc_base_ptr)

    # Compute the base pointers for the output based on the destination index
    Out_base_ptr += dest_idx * Out_seq_stride
    Out_scale_base_ptr += dest_idx * Out_scale_seq_stride

    # Quantize and store the data
    for i in range(0, hidden_dim, BLOCK_SIZE):
        # Load the data from K
        k_data = tl.load(K_base_ptr + i * K_hidden_stride, mask=i + tl.arange(0, BLOCK_SIZE) < hidden_dim, other=0.0)

        # Calculate the scale
        max_val = tl.max(k_data, axis=0)
        min_val = tl.min(k_data, axis=0)
        scale = (max_val - min_val) / 255.0

        # Quantize the data to int8
        quantized_data = tl.cast(tl.round(k_data / scale), tl.int8)

        # Store the quantized data and scale
        tl.store(Out_base_ptr + i * Out_hidden_stride, quantized_data, mask=i + tl.arange(0, BLOCK_SIZE) < hidden_dim)
        tl.store(Out_scale_base_ptr + i, scale, mask=i + tl.arange(0, BLOCK_SIZE) < hidden_dim)

import triton
import triton.runtime

def destindex_copy_quantize_kv(K, Out, Out_scale, Dest_loc):
    # Get the shape and strides of the input tensors
    batch_size, num_heads, seq_len, hidden_dim = K.shape
    K_batch_stride, K_head_stride, K_seq_stride, K_hidden_stride = K.strides
    Out_batch_stride, Out_head_stride, Out_seq_stride, Out_hidden_stride = Out.strides
    Out_scale_batch_stride, Out_scale_head_stride, Out_scale_seq_stride = Out_scale.strides
    Dest_loc_batch_stride, Dest_loc_head_stride, Dest_loc_seq_stride = Dest_loc.strides

    # Define the grid dimensions
    grid = (batch_size, num_heads, seq_len)

    # Launch the kernel
    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K.data_ptr(),  # Pointer to the input tensor K
        Out.data_ptr(),  # Pointer to the output tensor Out
        Out_scale.data_ptr(),  # Pointer to the output tensor Out_scale
        Dest_loc.data_ptr(),  # Pointer to the destination indices tensor Dest_loc
        K_batch_stride,  # Stride for batch dimension in K
        K_head_stride,  # Stride for head dimension in K
        K_seq_stride,  # Stride for sequence dimension in K
        K_hidden_stride,  # Stride for hidden dimension in K
        Out_batch_stride,  # Stride for batch dimension in Out
        Out_head_stride,  # Stride for head dimension in Out
        Out_seq_stride,  # Stride for sequence dimension in Out
        Out_hidden_stride,  # Stride for hidden dimension in Out
        Out_scale_batch_stride,  # Stride for batch dimension in Out_scale
        Out_scale_head_stride,  # Stride for head dimension in Out_scale
        Out_scale_seq_stride,  # Stride for sequence dimension in Out_scale
        Dest_loc_batch_stride,  # Stride for batch dimension in Dest_loc
        Dest_loc_head_stride,  # Stride for head dimension in Dest_loc
        Dest_loc_seq_stride,  # Stride for sequence dimension in Dest_loc
        batch_size,  # Batch size
        num_heads,  # Number of heads
        seq_len,  # Sequence length
        hidden_dim,  # Hidden dimension
        BLOCK_SIZE=32,  # Block size for parallelism
    )
