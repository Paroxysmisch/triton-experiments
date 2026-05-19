import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_softmax(
    logits_ptr,  # [B, H, N]
    output_ptr,  # [B, H, N]
    lengths_ptr, # [B]
    B,           # total batch size
    H,           # number of heads
    N,           # maximum sequence length
    stride_b,    # stride along batch dimension in logits
    stride_h,    # stride along head dimension in logits
    stride_n,    # stride along seq dimension in logits
    BLOCK_SIZE: tl.constexpr
):
    # program ids
    b_id = tl.program_id(0)  # batch index
    h_id = tl.program_id(1)  # head index

    # offset to start of this (b, h)
    # each block works on one (batch, head) pair, then processes tokens in [0..BLOCK_SIZE]
    b_offset = b_id
    h_offset = h_id

    # load valid length
    seq_len = tl.load(lengths_ptr + b_offset)

    # create a range of token indices
    t_range = tl.arange(0, BLOCK_SIZE)
    # compute the absolute token index
    t_index = t_range
    # create mask to handle padding
    mask = t_index < seq_len

    # base pointer for 1D slice [N] we are working on
    logits_base = logits_ptr + b_offset * stride_b + h_offset * stride_h
    output_base = output_ptr + b_offset * stride_b + h_offset * stride_h

    # load logits
    # mask out positions >= seq_len by setting them to -inf
    logits = tl.load(logits_base + t_index * stride_n, mask=mask, other=-float('inf'))

    # compute max
    max_logits = tl.maximum(tl.max(logits, axis=0), 0)  # max of valid range
    # subtract max and exponentiate
    logits = logits - max_logits
    num = tl.exp(logits)
    # sum
    denom = tl.sum(num, axis=0)
    # normalize
    out = num / denom

    # store results
    tl.store(output_base + t_index * stride_n, out, mask=mask)


def token_softmax_fwd(logits, output, lengths):
    """
    logits:  [B, H, N]
    output:  [B, H, N]
    lengths: [B]
    """
    B, H, N = logits.size()
    # strides in memory
    stride_b = logits.stride(0)
    stride_h = logits.stride(1)
    stride_n = logits.stride(2)

    # pick an appropriate block size
    BLOCK_SIZE = 128
    grid = (B, H)

    _fwd_kernel_token_softmax[grid](
        logits, 
        output, 
        lengths,
        B, H, N,
        stride_b, stride_h, stride_n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,    # can be tuned
        num_stages=2    # can be tuned
    )
