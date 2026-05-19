import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_softmax(
    logit_ptr,
    output_ptr,
    max_logit_ptr,
    lengths_ptr,
    stride_logit,
    stride_output,
    stride_lengths,
    batch_size,
    num_heads,
    max_seq_len,
    block_size,
    num_warps,
    num_stages,
):
    """
    Compute the softmax of token logits with variable sequence lengths.
    """
    pid = tl.program_id(axis=0)
    num_tokens = batch_size * num_heads * max_seq_len
    num_blocks = tl.cdiv(num_tokens, block_size)

    # Compute the start and end indices for the current block
    block_start = pid * block_size
    block_end = min(block_start + block_size, num_tokens)

    # Load the lengths for the current block
    lengths = tl.load(lengths_ptr + pid * stride_lengths)

    for token in range(block_start, block_end):
        # Compute the sequence index and head index
        seq_idx = token // (num_heads * max_seq_len)
        head_idx = (token // max_seq_len) % num_heads
        seq_offset = seq_idx * stride_logit
        head_offset = head_idx * max_seq_len
        token_offset = token % max_seq_len

        # Load the logits for the current token
        logits = tl.load(logit_ptr + seq_offset + head_offset + token_offset * stride_logit)

        # Compute the maximum logit for the current sequence and head
        max_logit = tl.max(logits[:lengths[seq_idx]])

        # Subtract the maximum logit from the logits to improve numerical stability
        logits -= max_logit

        # Compute the exponentiated logits
        exp_logits = tl.exp(logits)

        # Compute the sum of the exponentiated logits
        exp_sum = tl.sum(exp_logits[:lengths[seq_idx]])

        # Normalize the logits
        output = exp_logits / exp_sum

        # Store the output in the output tensor
        tl.store(output_ptr + seq_offset + head_offset + token_offset * stride_output, output)

@triton.jit
def token_softmax_fwd(
    logit_ptr,
    output_ptr,
    max_logit_ptr,
    lengths_ptr,
    stride_logit,
    stride_output,
    stride_lengths,
    batch_size,
    num_heads,
    max_seq_len,
    block_size,
    num_warps,
    num_stages,
):
    """
    Wrapper function to orchestrate the execution of the softmax kernel.
    """
    num_tokens = batch_size * num_heads * max_seq_len
    num_blocks = tl.cdiv(num_tokens, block_size)

    for pid in range(num_blocks):
        _fwd_kernel_token_softmax[
            num_blocks,  # grid size
            block_size,  # block size
            num_warps,   # number of warps per block
            num_stages   # number of stages for pipeline parallelism
        ](
            logit_ptr,
            output_ptr,
            max_logit_ptr,
            lengths_ptr,
            stride_logit,
            stride_output,
            stride_lengths,
            batch_size,
            num_heads,
            max_seq_len,
            block_size,
            num_warps,
            num_stages,
        )

# Example usage:
# logit_ptr = ...  # Pointer to the input logits tensor
# output_ptr = ...  # Pointer to the output tensor
# max_logit_ptr = ...  # Pointer to the max logit tensor
# lengths_ptr = ...  # Pointer to the lengths tensor
# stride_logit = ...  # Stride of the logits tensor
# stride_output = ...  # Stride of the output tensor
# stride_lengths = ...  # Stride of the lengths tensor
# batch_size = ...  # Batch size
# num_heads = ...  # Number of attention heads
# max_seq_len = ...  # Maximum sequence length
# block_size = ...  # Block size for parallelism
# num_warps = ...  # Number of warps per block
# num_stages = ...  # Number of stages for pipeline parallelism

# token_softmax_fwd(
#     logit_ptr,
#     output_ptr,
#     max_logit_ptr,
#     lengths_ptr,
#     stride_logit,
#     stride_output,
#     stride_lengths,
#     batch_size,
#     num_heads,
#     max_seq_len,
#     block_size,
#     num_warps,
#     num_stages,
# )
