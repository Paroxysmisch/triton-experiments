import triton
import triton.language as tl

@triton.jit
def _bgmv_shrink_kernel(
    lora_indices_ptr,
    input_ptr,
    lora_ptr,
    out_ptr,
    batch_count,
    N,
    K,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    scaling,
    ld_input: tl.constexpr,
    ld_lora: tl.constexpr,
    ld_out: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    bid = tl.block_id(axis=0)
    n = tl.program_id(axis=1) * BLOCK_N + tl.masked_off(tl.program_id(axis=0) * BLOCK_K, pid >= batch_count)
    k_start = SPLIT_K * pid
    k_end = min(K, SPLIT_K * (pid + 1))
    ld_k = ld_lora // N

    M = K + 1 - k_start
    ld_out_row = ld_out // N
    out = tl.zeros((M,), dtype=tl.float32)

    for k in range(k_start, k_end):
        l = ld_k * (k - k_start)
        if n < N:
            in_row = input_ptr + n * ld_input
            lora_row = lora_ptr + l
            v = tl.dot(in_row[k:k + BLOCK_K], lora_row[k:k + BLOCK_K])
            tl.atomic_add(out, (k - k_start,), scaling * v)

    if n < N:
        out_row = out_ptr + n * ld_out_row
        for k in range(k_start, k_end):
            l = ld_k * (k - k_start)
            tl.store(out_row, out[k - k_start], ld_out_row * (k - k_start))
            

def _bgmv_shrink(
    lora_indices: torch.Tensor,
    input: torch.Tensor,
    lora: torch.Tensor,
    out: torch.Tensor,
    scaling: float,
    max_batch_count: int,
    max_grid: Tuple[Optional[int], Optional[int], Optional[int]] = None,
):
    assert input.is_contiguous()
    assert lora.is_contiguous()
    assert out.is_contiguous()

    ld_input = input.stride(0)
    ld_lora = lora.stride(0)
    ld_out = out.stride(0)
    N = lora.size(0)
    K = lora.size(1)
    BLOCK_N = 2 ** tl.ceil(tl.log2(N))
    batch_count = min(lora_indices.size(0), max_batch_count)

    grid = (tl.ceil(N / BLOCK_N), tl.ceil(K / SPLIT_K), 1)
    if max_grid is not None:
        grid = (min(grid[0], max_grid[0]), min(grid[1], max_grid[1]), 1)

    _bgmv_shrink_kernel[grid](
        lora_indices.data_ptr(),
        input.data_ptr(),
        lora.data_ptr(),
        out.data_ptr(),
        batch_count,
        N,
        K,
        BLOCK_N,
        SPLIT_K,
        scaling,
        ld_input,
        ld_lora,
        ld_out,
    )
    return out
