import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_token_softmax(
    logits_ptr, b_start_loc_ptr, b_seqlen_ptr, prob_out_ptr,
    stride_logits_m, stride_logits_h, stride_logits_n,
    stride_prob_m, stride_prob_h, stride_prob_n,
    BLOCK_SIZE: tl.constexpr
):
    # Get the program id
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Calculate the start index and sequence length for the current batch and head
    start_loc = tl.load(b_start_loc_ptr + pid_m)
    seqlen = tl.load(b_seqlen_ptr + pid_m)

    # Create offsets for logits and output probabilities
    offsets = start_loc + tl.arange(0, BLOCK_SIZE)
    offsets = tl.where(offsets < start_loc + seqlen, offsets, start_loc + seqlen - 1)

    # Load logits for the current sequence
    logits = tl.load(logits_ptr + pid_m * stride_logits_m + pid_h * stride_logits_h + offsets * stride_logits_n, mask=offsets < start_loc + seqlen, other=-float('inf'))

    # Compute the max logits for numerical stability
    max_logits = tl.max(logits, axis=0)

    # Subtract max_logits, exponentiate and sum for normalization
    logits = logits - max_logits
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0)

    # Compute probabilities
    probs = exp_logits / sum_exp_logits

    # Store the probabilities
    tl.store(prob_out_ptr + pid_m * stride_prob_m + pid_h * stride_prob_h + offsets * stride_prob_n, probs, mask=offsets < start_loc + seqlen)

def token_softmax_fwd(logits, b_start_loc, b_seqlen, prob_out):
    # Ensure input tensors are contiguous
    logits = logits.contiguous()
    b_start_loc = b_start_loc.contiguous()
    b_seqlen = b_seqlen.contiguous()
    prob_out = prob_out.contiguous()

    # Get the shapes and strides
    batch_size, num_heads, max_seqlen = logits.shape
    stride_logits_m, stride_logits_h, stride_logits_n = logits.stride()
    stride_prob_m, stride_prob_h, stride_prob_n = prob_out.stride()

    # Launch the Triton kernel
    grid = (batch_size, num_heads)
    BLOCK_SIZE = 128  # Adjust based on your hardware's capabilities

    _fwd_kernel_token_softmax[grid](
        logits, b_start_loc, b_seqlen, prob_out,
        stride_logits_m, stride_logits_h, stride_logits_n,
        stride_prob_m, stride_prob_h, stride_prob_n,
        BLOCK_SIZE=BLOCK_SIZE
    )

# Example usage
batch_size = 2
num_heads = 4
max_seqlen = 128

logits = torch.randn((batch_size, num_heads, max_seqlen), device='cuda', dtype=torch.float32)
b_start_loc = torch.tensor([0, 64], device='cuda', dtype=torch.int32)
b_seqlen = torch.tensor([64, 64], device='cuda', dtype=torch.int32)
prob_out = torch.empty_like(logits)

token_softmax_fwd(logits, b_start_loc, b_seqlen, prob_out)
