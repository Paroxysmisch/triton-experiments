import triton
import triton.language as tl

from .utils import chunks, split_into_groups, define_chunk_size

try:
    from ..ops.rms_norm import RMSNorm
except ImportError:
    from silu_norm import RMSNorm

try:
    from ..utils import get_autotune_triton_configs
except:
    def get_autotune_triton_configs():
        return [{"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 32, "num_warps": 4},]*4

@triton.jit
def ff_llama(
    x,
    w1,
    w3,
    rms_w,
    y,
    stride_x_rows,
    BLOCK_SIZE_SEQ,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    act_in_place,
    diagonal_offset_m,
    diagonal_offset_n,
    show_chunks,
    ALLOW_TF32,
    X_GROUP_SIZE,
    use_fp8,
    **meta
):
    N_CTX = tl.program_id(0)
    N_ROW = tl.program_id(1)
    # chunk_size_seq determine the sequence length within a chunk
    # i.e., chunk_size_seq=16 (for Llama 256 context) or 32 (for Llama 512 context)
    # If you want to make it works with larger context like 1024 you may need to increase the chunk size
    # But be aware that it will increase memory usage and require more GPU ram
    chunk_size_seq = min(BLOCK_SIZE_SEQ, tl.cdiv(meta["seq_len"], tl.cdiv(N_ROW, 2) * 2))
    chunk_size_m = min(BLOCK_SIZE_M, tl.cdiv(meta["n_heads_ngpus"], 2))
    chunk_size_k = min(BLOCK_SIZE_K, 2048 // 8)  # => 2048 // xxx

    ALLOW_TF32 = meta["allow_tf32"]

    x_ptr = x + N_CTX * stride_x_rows
    y_ptr = y + N_CTX * meta["y_stride"]
    w1_ptr = w1 + N_ROW * meta["w1_stride"]
    w3_ptr = w3 + N_ROW * meta["w3_stride"]

    # w1 for diagonal element
    w1 += diagonal_offset_m * meta["w1_stride"]

    if act_in_place:
        y_ptr = x_ptr
        diagonal_offset_m = 0
    else:
        w1 += diagonal_offset_n * meta["w1_stride"]
        w3 += diagonal_offset_n * meta["w3_stride"]

    group_id = tl.program_id(2)
    offs_m = group_id * X_GROUP_SIZE + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (
        (offs_m[:, None] * stride_x_rows + offs_n[None, :] * meta["x_stride"]) // 8
    )  # chunk_size_m*chunk_size_n

    w1_ptrs = (
        w1_ptr
        + (offs_n[:, None] * meta["w1_stride"] + offs_k[None, :] * meta["w1_ld"]) // 8
    )  # chunk_size_n*chunk_size_k
    w3_ptrs = (
        w3_ptr
        + (offs_m[:, None] * meta["w3_stride"] + offs_k[None, :] * meta["w3_ld"]) // 8
    )  # chunk_size_m*chunk_size_k

    # We unroll by 2 since each "row" of w is NxK and we process it in chunks of N
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # We use smaller type for fp8 calculation which is much faster
    acc1_f32 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2_f32 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    rm = 0
    for k in range(0, meta["n_elements"] // chunk_size_seq):
        mask_x_loaded = tl.reshape(tl.where(k * chunk_size_seq + tl.arange(0, chunk_size_seq) < meta["seq_len"], 1, 0), (chunk_size_seq,))
        mask_k_loaded = tl.reshape(tl.where(tl.arange(0, chunk_size_seq) < chunk_size_seq, 1, 0), (chunk_size_seq,))

        x_chunk = tl.load(x_ptrs, mask=mask_x_loaded[:, None] * mask_x_loaded[None, :], other=0, cache_modifier=".cg")
        w1_chunk = tl.load(w1_ptrs, mask=mask_k_loaded[:, None] * mask_x_loaded[None, :], other=0, cache_modifier=".cg")
        w3_chunk = tl.load(w3_ptrs, mask=mask_k_loaded[None, :] * mask_x_loaded[:, None], other=0, cache_modifier=".cg")

        if tl.core._is_mps():
            acc1 += tl.dot(w1_chunk, x_chunk, allow_tf32=ALLOW_TF32).to(tl.float32)
            acc2 += w3_chunk * x_chunk
        else:
            acc1 += tl.dot(w1_chunk, x_chunk, allow_tf32=ALLOW_TF32)
            acc2 += w3_chunk * x_chunk

        if use_fp8:
            acc1_f32 += tl.dot(w1_chunk, x_chunk)
            acc2_f32 += w3_chunk * x_chunk

        x_ptrs += chunk_size_seq * meta["x_stride"] // 8
        w1_ptrs += chunk_size_seq * meta["w1_stride"] // 8
        w3_ptrs += chunk_size_seq * meta["w3_stride"] // 8

        p = chunk_size_seq * k
        q = p + chunk_size_seq
        cm = min(q, meta["seq_len"]) - max(p, 0)

        if cm.to(tl.int32) == 0:
            continue

        mask1 = (k * chunk_size_seq) % meta["group_start_token"] < meta["n_ctx_in_chunk"]
        mask0 = offs_m < meta["n_heads_ngpus"]
        rms_once = diagonal_offset_m == 0 and diagonal_offset_n == 0 and mask1 and mask0

        if use_fp8:
            x_chunk = (x_chunk * (127.0 / 16.0)).to(tl.int8, bitcast=True)
            w1_chunk = (w1_chunk * (127.0 / 16.0)).to(tl.int8,
