import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    Prob,  # Probability matrix
    V,     # Value tensor
    Out,   # Output tensor
    Req_to_tokens,  # Mapping requests to tokens
    req_batch_stride,  # Stride for batch in Req_to_tokens
    req_head_stride,   # Stride for head in Req_to_tokens
    req_token_stride,  # Stride for token in Req_to_tokens
    prob_batch_stride,  # Stride for batch in Prob
    prob_head_stride,   # Stride for head in Prob
    prob_token_stride,  # Stride for token in Prob
    v_batch_stride,     # Stride for batch in V
    v_head_stride,      # Stride for head in V
    v_token_stride,     # Stride for token in V
    out_batch_stride,   # Stride for batch in Out
    out_head_stride,    # Stride for head in Out
    out_token_stride,   # Stride for token in Out
    BLOCK: tl.constexpr,  # Block size
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_token = tl.program_id(2) * BLOCK

    # Load the request to token mapping
    req_to_token = tl.load(Req_to_tokens + cur_batch * req_batch_stride + cur_head * req_head_stride + cur_token * req_token_stride)

    # Initialize the accumulator
    acc = tl.zeros((BLOCK, V.shape[-1]), dtype=tl.float32)

    # Iterate over the token sequence in blocks
    for block_start in range(0, req_to_token, BLOCK):
        p_value = tl.load(Prob + cur_batch * prob_batch_stride + cur_head * prob_head_stride + block_start * prob_token_stride + tl.arange(0, BLOCK)[:, None])
        v_value = tl.load(V + cur_batch * v_batch_stride + cur_head * v_head_stride + block_start * v_token_stride + tl.arange(0, V.shape[-1])[None, :])

        # Compute the attention
        acc += p_value @ v_value

    # Cast the result to the output data type and store it
    out = acc.to(Out.dtype.element_ty)
    tl.store(Out + cur_batch * out_batch_stride + cur_head * out_head_stride + cur_token * out_token_stride + tl.arange(0, BLOCK)[:, None], out)

import torch
from torch import Tensor
from triton import cdiv, launch_kernel

@torch.no_grad()
def token_att_fwd2(
    Prob: Tensor,  # Probability matrix
    V: Tensor,     # Value tensor
    Out: Tensor,   # Output tensor
    Req_to_tokens: Tensor,  # Mapping requests to tokens
    req_batch_stride: int,  # Stride for batch in Req_to_tokens
    req_head_stride: int,   # Stride for head in Req_to_tokens
    req_token_stride: int,  # Stride for token in Req_to_tokens
    prob_batch_stride: int,  # Stride for batch in Prob
    prob_head_stride: int,   # Stride for head in Prob
    prob_token_stride: int,  # Stride for token in Prob
    v_batch_stride: int,     # Stride for batch in V
    v_head_stride: int,      # Stride for head in V
    v_token_stride: int,     # Stride for token in V
    out_batch_stride: int,   # Stride for batch in Out
    out_head_stride: int,    # Stride for head in Out
    out_token_stride: int,   # Stride for token in Out
    BLOCK: int = 128,  # Block size
):
    # Compute the number of heads for key-value pairs
    kv_group_num = V.shape[1] // Prob.shape[1]

    # Determine the grid dimensions
    grid = (
        cdiv(Prob.shape[0], BLOCK),  # Number of batches
        Prob.shape[1],              # Number of heads
        cdiv(Prob.shape[2], BLOCK)  # Number of token blocks
    )

    # Launch the Triton kernel
    launch_kernel(_fwd_kernel_token_att2, grid, [
        Prob.data_ptr(), V.data_ptr(), Out.data_ptr(),
        Req_to_tokens.data_ptr(),
        req_batch_stride, req_head_stride, req_token_stride,
        prob_batch_stride, prob_head_stride, prob_token_stride,
        v_batch_stride, v_head_stride, v_token_stride,
        out_batch_stride, out_head_stride, out_token_stride,
        BLOCK
    ])
