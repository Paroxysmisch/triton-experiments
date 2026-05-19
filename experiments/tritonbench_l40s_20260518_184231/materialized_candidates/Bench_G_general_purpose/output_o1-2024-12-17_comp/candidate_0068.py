import triton
import triton.language as tl
import torch

@triton.jit
def _multinomial_sampling_kernel(
    scores_ptr, seeds_ptr, output_ptr,
    batch_size, num_tokens,
    stride_b, stride_t,
    BLOCK: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_b = tl.program_id(0)
    if pid_b >= batch_size:
        return

    seed = tl.load(seeds_ptr + pid_b).to(tl.uint32)
    seed ^= seed << 13
    seed ^= seed >> 17
    seed ^= seed << 5
    random_u32 = seed
    random_f = random_u32.to(tl.float32) / float(2**32)

    offset = 0
    cumsum_offset = 0.0
    found_index = -1

    while offset < num_tokens:
        offs = offset + tl.arange(0, BLOCK_N)
        mask = offs < num_tokens
        block_scores = tl.load(scores_ptr + pid_b * stride_b + offs * stride_t, mask=mask, other=0.0)
        partial_sums = tl.zeros([BLOCK_N], dtype=tl.float32)
        running_sum = 0.0
        for i in range(BLOCK_N):
            running_sum += block_scores[i]
            partial_sums[i] = running_sum
        for i in range(BLOCK_N):
            if found_index < 0:
                if random_f <= cumsum_offset + partial_sums[i]:
                    found_index = offset + i
        cumsum_offset += partial_sums[BLOCK_N - 1]
        offset += BLOCK_N

    if found_index < 0:
        found_index = num_tokens - 1

    tl.store(output_ptr + pid_b, found_index)

def multinomial_sampling(scores):
    batch_size, num_tokens = scores.shape
    seeds = torch.randint(0, 2**31, (batch_size,), device=scores.device, dtype=torch.int32)
    output = torch.empty((batch_size,), dtype=torch.int32, device=scores.device)
    grid = (batch_size,)
    _multinomial_sampling_kernel[grid](
        scores, seeds, output,
        batch_size, num_tokens,
        scores.stride(0), scores.stride(1),
        BLOCK=8, BLOCK_N=128
    )
    return output
