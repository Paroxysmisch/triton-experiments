import torch
import triton
import triton.language as tl
from deepspeed.accelerator import get_accelerator

@triton.jit
def _fwd_kernel_token_softmax(
    logits_ptr,
    logits_max_ptr,
    logits_sum_ptr,
    logits_exp_sum_ptr,
    lse_ptr,
    seq_lens_ptr,
    attention_mask_ptr,
    output_ptr,
    batch_size,
    max_seq_len,
    num_heads,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    logits_ptr += pid_b * max_seq_len * num_heads + pid_h * max_seq_len
    seq_lens_ptr += pid_b
    attention_mask_ptr += pid_b * max_seq_len
    output_ptr += pid_b * max_seq_len * num_heads + pid_h * max_seq_len

    offs_m = tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, BLOCK_SIZE)
    logits_mask = offs_n < seq_len

    seq_len = tl.load(seq_lens_ptr)

    attention_mask = tl.load(attention_mask_ptr + offs_n, mask=offs_n < seq_len, other=-float("inf"))

    logits = tl.load(logits_ptr + offs_m[:, None] * num_heads + offs_n[None, :],
                     mask=offs_m[:, None] * num_heads + offs_n[None, :] < seq_len * num_heads,
                     other=-float("inf"))

    logits = tl.where(logits_mask[None, :] & (offs_m[:, None] < seq_len), logits, -float("inf"))

    logits_max = tl.max(logits, axis=1)
    tl.store(logits_max_ptr + offs_m, logits_max)

    logits = logits - logits_max[:, None]
    logits_exp = tl.exp(logits)
    tl.store(logits_exp_sum_ptr + offs_m, tl.sum(logits_exp, axis=1))

    lse = tl.log(tl.sum(logits_exp, axis=1))
    tl.store(lse_ptr + offs_m, lse)

    output = logits_exp / (tl.exp(lse[:, None] - logits) + 1e-6)
    output = tl.where(attention_mask[None, :] != -float("inf"), output, 0)
    tl.store(output_ptr + offs_m[:, None] * num_heads + offs_n[None, :],
             output,
             mask=offs_m[:, None] * num_heads + offs_n[None, :] < seq_len * num_heads)
    return

@torch.no_grad()
def token_softmax_fwd(logits, seq_len, attention_mask, logits_max, logits_sum, logits_exp_sum, lse):
    batch_size, max_seq_len, num_heads = logits.shape
    seq_len = torch.tensor(seq_len, dtype=torch.int32, device=logits.device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.bool, device=logits.device)
    logits_max = torch.empty_like(seq_len, dtype=torch.float32, device=get_accelerator().current_device())
    logits_sum = torch.empty_like(seq_len, dtype=torch.float32, device=get_accelerator().current_device())
    logits_exp_sum = torch.empty_like(seq_len, dtype=torch.float32, device=get_accelerator().current_device())
    lse = torch.empty_like(seq_len, dtype=torch.float32, device=get_accelerator().current_device())

    BLOCK_SIZE = triton.next_power_of_2(max_seq_len)
    num_warps = 4
    if BLOCK_SIZE > 2047:
        num_warps = 8
    elif BLOCK_SIZE > 4095:
        num_warps = 16

    _fwd_kernel_token_softmax[(batch_size, num_heads)](logits, logits_max, logits_sum, logits_exp_sum, lse,
                                                       seq_len, attention_mask, logits, batch_size, max_seq_len,
                                                       num_heads, seq_len, num_warps=num_warps,
                                                       BLOCK_SIZE=BLOCK_SIZE)
    return logits_max, logits_sum, logits_exp_sum, lse
