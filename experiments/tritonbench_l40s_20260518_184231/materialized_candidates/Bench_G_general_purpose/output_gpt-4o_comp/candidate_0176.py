import triton
import triton.language as tl
import torch

# Triton kernel for token softmax forward pass
@triton.jit
def _fwd_kernel_token_softmax(
    logits_ptr,       # Pointer to the logits input tensor
    output_ptr,       # Pointer to the output tensor
    mask_ptr,         # Pointer to the mask tensor (optional, can be None)
    seq_lengths_ptr,  # Pointer to the sequence lengths tensor
    batch_size: tl.constexpr,  # Number of batches
    num_heads: tl.constexpr,   # Number of attention heads
    max_seq_len: tl.constexpr, # Maximum sequence length
    BLOCK_SIZE: tl.constexpr   # Block size for parallelism
):
    # Block indices
    batch_idx = tl.program_id(0)  # Each block handles a batch
    head_idx = tl.program_id(1)   # Each block handles a head

    # Start of the sequence for this batch and head
    start_idx = (batch_idx * num_heads + head_idx) * max_seq_len

    # Load sequence length for this batch
    seq_len = tl.load(seq_lengths_ptr + batch_idx)

    # Compute the range of tokens this block will handle
    offsets = tl.arange(0, BLOCK_SIZE)

    # Compute the mask for valid tokens
    valid_mask = offsets < seq_len

    # Load logits for the current batch and head
    logits = tl.load(logits_ptr + start_idx + offsets, mask=valid_mask, other=-float('inf'))

    # Step 1: Compute the maximum logit value for numerical stability
    max_logits = tl.reduce.max(logits, axis=0)

    # Step 2: Subtract the maximum and exponentiate
    logits = logits - max_logits
    exp_logits = tl.exp(logits)

    # Step 3: Compute the sum of exponentiated logits
    sum_exp_logits = tl.reduce.sum(exp_logits, axis=0)

    # Step 4: Compute softmax
    softmax = exp_logits / sum_exp_logits

    # Write the results back to the output tensor
    tl.store(output_ptr + start_idx + offsets, softmax, mask=valid_mask)

# Wrapper function for launching the Triton kernel
def token_softmax_fwd(logits, seq_lengths, mask=None):
    """
    Computes the softmax of token logits with variable sequence lengths.
    
    Args:
        logits (torch.Tensor): Input tensor of shape [batch_size, num_heads, max_seq_len].
        seq_lengths (torch.Tensor): Sequence lengths tensor of shape [batch_size].
        mask (torch.Tensor, optional): Mask tensor of shape [batch_size, num_heads, max_seq_len].
    
    Returns:
        torch.Tensor: Softmax output of shape [batch_size, num_heads, max_seq_len].
    """
    assert logits.ndim == 3, "Logits must be a 3D tensor [batch_size, num_heads, max_seq_len]"
    batch_size, num_heads, max_seq_len = logits.shape

    # Allocate output tensor
    output = torch.empty_like(logits)

    # Flatten the input tensors for easier indexing in Triton
    logits_flat = logits.view(-1)
    output_flat = output.view(-1)
    seq_lengths_flat = seq_lengths.view(-1)

    # Determine the number of blocks
    BLOCK_SIZE = 128  # You can tune this value for performance
    num_warps = 4     # Number of warps for parallelism, tune this as well

    # Launch the Triton kernel
    grid = (batch_size, num_heads)  # One block per batch and head
    triton.kernel(
        _fwd_kernel_token_softmax,
        grid=grid,
        num_warps=num_warps,
        num_stages=2,
        kwargs=dict(
            logits_ptr=logits_flat,
            output_ptr=output_flat,
            mask_ptr=mask,
            seq_lengths_ptr=seq_lengths_flat,
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            BLOCK_SIZE=BLOCK_SIZE
        )
    )

    return output
