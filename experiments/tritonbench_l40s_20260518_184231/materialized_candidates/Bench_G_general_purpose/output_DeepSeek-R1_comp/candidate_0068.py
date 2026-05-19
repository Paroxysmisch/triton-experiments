import torch
import triton
import triton.language as tl

@triton.jit
def _multinomial_sampling_kernel(
    output_ptr,
    scores_ptr,
    seed_ptr,
    batch_size,
    num_tokens,
    stride_scores_batch,
    stride_scores_token,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= batch_size:
        return

    seed = tl.load(seed_ptr + pid)
    u = tl.rand(seed, tl.zeros((1,), tl.int32))[0]

    current_total = 0.0
    output_index = num_tokens  # Initialize with invalid index

    num_blocks = tl.cdiv(num_tokens, BLOCK_N)
    
    for block_idx in range(num_blocks):
        block_start = block_idx * BLOCK_N
        offsets = block_start + tl.arange(0, BLOCK_N)
        mask = offsets < num_tokens

        scores = tl.load(
            scores_ptr + pid * stride_scores_batch + offsets * stride_scores_token,
            mask=mask,
            other=0.0,
        )

        cum_sum = tl.associative_scan(scores, 0, tl.math.add)
        sum_block = cum_sum[-1]
        new_total = current_total + sum_block

        if new_total >= u:
            residual = u - current_total
            mask_in_block = cum_sum >= residual
            int_mask = tl.where(mask_in_block, 1, 0)
            pos_in_block = tl.argmax(int_mask, axis=0)

            if tl.max(int_mask) == 1:
                output_index = block_start + pos_in_block
                break
            current_total = new_total
        else:
            current_total = new_total

    # Fallback to last index if not found (shouldn't happen for valid inputs)
    output_index = tl.where(output_index == num_tokens, num_tokens - 1, output_index)
    tl.store(output_ptr + pid, output_index)

def multinomial_sampling(scores: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    assert scores.dim() == 2, "Scores must be a 2D tensor"
    batch_size, num_tokens = scores.shape
    output = torch.empty(batch_size, dtype=torch.int64, device=scores.device)

    BLOCK_N = 128
    grid = (batch_size,)

    _multinomial_sampling_kernel[grid](
        output,
        scores,
        seed,
        batch_size,
        num_tokens,
        scores.stride(0),
        scores.stride(1),
        BLOCK_N=BLOCK_N,
    )
    return output
