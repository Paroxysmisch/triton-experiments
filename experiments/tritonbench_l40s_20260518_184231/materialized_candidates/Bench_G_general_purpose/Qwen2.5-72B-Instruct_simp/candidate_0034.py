import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    prob_ptr,  # Pointer to the probability tensor (B, H, S, W)
    value_ptr,  # Pointer to the value tensor (B, H, S, D)
    output_ptr,  # Pointer to the output tensor (B, H, S, D)
    B,  # Batch size
    H,  # Number of heads
    S,  # Sequence length
    D,  # Dimension of the value tensor
    W,  # Window size
    stride_prob_b,  # Stride for batch in prob tensor
    stride_prob_h,  # Stride for head in prob tensor
    stride_prob_s,  # Stride for sequence in prob tensor
    stride_prob_w,  # Stride for window in prob tensor
    stride_value_b,  # Stride for batch in value tensor
    stride_value_h,  # Stride for head in value tensor
    stride_value_s,  # Stride for sequence in value tensor
    stride_value_d,  # Stride for dimension in value tensor
    stride_output_b,  # Stride for batch in output tensor
    stride_output_h,  # Stride for head in output tensor
    stride_output_s,  # Stride for sequence in output tensor
    stride_output_d,  # Stride for dimension in output tensor
    BLOCK_SIZE_S: tl.constexpr,  # Block size for sequence
    BLOCK_SIZE_D: tl.constexpr,  # Block size for dimension
    BLOCK_SIZE_W: tl.constexpr  # Block size for window
):
    pid = tl.program_id(axis=0)
    bid = pid // (H * S)
    hid = (pid % (H * S)) // S
    sid = (pid % (H * S)) % S

    # Offsets for batch, head, and sequence
    prob_offset = bid * stride_prob_b + hid * stride_prob_h + sid * stride_prob_s
    value_offset = bid * stride_value_b + hid * stride_value_h + sid * stride_value_s
    output_offset = bid * stride_output_b + hid * stride_output_h + sid * stride_output_s

    # Load the probability and value tensors
    prob = tl.load(prob_ptr + prob_offset + tl.arange(0, BLOCK_SIZE_W), mask=tl.arange(0, BLOCK_SIZE_W) < W, other=0.0)
    value = tl.load(value_ptr + value_offset + tl.arange(0, BLOCK_SIZE_D), mask=tl.arange(0, BLOCK_SIZE_D) < D, other=0.0)

    # Compute the output
    output = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    for w in range(W):
        output += prob[w] * value

    # Store the output
    tl.store(output_ptr + output_offset + tl.arange(0, BLOCK_SIZE_D), output, mask=tl.arange(0, BLOCK_SIZE_D) < D)

import torch

def token_att_fwd2(prob, value, window_size):
    B, H, S, W = prob.shape
    B, H, S, D = value.shape

    # Allocate output tensor
    output = torch.empty((B, H, S, D), device=prob.device, dtype=prob.dtype)

    # Define grid and block dimensions
    grid = (B * H * S,)
    block = (128,)  # Adjust block size as needed

    # Launch the kernel
    _fwd_kernel_token_att2[grid, block](
        prob, value, output,
        B, H, S, D, W,
        prob.stride(0), prob.stride(1), prob.stride(2), prob.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE_S=1,  # Adjust block size for sequence
        BLOCK_SIZE_D=128,  # Adjust block size for dimension
        BLOCK_SIZE_W=window_size  # Adjust block size for window
    )

    return output
