import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_D": 32},
            num_stages=5,
            num_warps=2,
        ),
    ],
    key=["n_ctx_q", "n_ctx_k", "d_model"],
    prune_configs_by={
        "early_config_prune": triton.ops.matmul_perf_model.early_config_prune,
        "perf_model": triton.ops.matmul_perf_model.estimate_matmul_time,
        "top_k": 10,
    },
)
@triton.jit
def _score_kernel(
    q_ptr, k_ptr, m_ptr, out_ptr,
    n_ctx_q,
    n_ctx_k,
    d_model,
    stride_ctx_q, stride_ctx_k, stride_d,
    stride_out_q, stride_out_k,
    sm_scale: tl.float32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    window_size: tl.constexpr,
):
    pid = tl.program_id(0)

    # Determine the number of blocks in the grid
    grid_n = (n_ctx_k + BLOCK_N - 1) // BLOCK_N

    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # Compute the range of indices for the current block
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Compute the mask for the current block
    mask = (rm < n_ctx_q)[:, None] & (rn < n_ctx_k)[None, :]

    # Initialize the score matrix
    qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate through blocks of the d_model dimension and accumulate values into qk
    rd = tl.arange(0, BLOCK_D)

    q_ptr_block = q_ptr + (rm[:, None] * stride_ctx_q + rd[None, :] * stride_d)
    k_ptr_block = k_ptr + (rd[:, None] * stride_d + rn[None, :] * stride_ctx_k)

    for d_max_offset in range(d_model, 0, -BLOCK_D):
        q_block = tl.load(q_ptr_block, mask=rd[None, :] < d_max_offset, other=0.0)
        k_block = tl.load(k_ptr_block, mask=rd[:, None] < d_max_offset, other=0.0)

        qk += tl.dot(q_block, k_block, allow_tf32=True)

        q_ptr_block += BLOCK_D * stride_d
        k_ptr_block += BLOCK_D * stride_d

    # Apply the scaling factor
    qk *= sm_scale

    # Apply the mask if provided
    if m_ptr is not None:
        m_block = tl.load(m_ptr + (rm[:, None] * stride_out_q + rn[None, :] * stride_out_k), mask=mask, other=0.0)
        qk += m_block

    # Apply the sliding window mask if provided
    if window_size > 0:
        window_mask = (tl.abs(rm[:, None] - rn[None, :]) <= window_size)
        qk = tl.where(window_mask, qk, float('-inf'))

    # Store the result
    out_ptr_block = out_ptr + (rm[:, None] * stride_out_q + rn[None, :] * stride_out_k)
    tl.store(out_ptr_block, qk, mask=mask)

def get_score(query, key, mask=None, window_size=0):
    device = query.device

    # Handle non-contiguous inputs if necessary
    if query.stride(0) > 1 and query.stride(1) > 1:
        query = query.contiguous()
    if key.stride(0) > 1 and key.stride(1) > 1:
        key = key.contiguous()
    if mask is not None and (mask.stride(0) > 1 or mask.stride(1) > 1):
        mask = mask.contiguous()

    # Check constraints
    n_ctx_q, d_model = query.shape
    n_ctx_k, d_model_k = key.shape
    assert d_model == d_model_k, f"{query.shape=} {key.shape=}"

    # Allocate output
    out = torch.empty((n_ctx_q, n_ctx_k), device=device, dtype=query.dtype)

    # Stride along the d_model dimension
    stride_d = query.stride(1)
    assert stride_d == key.stride(1), f"{stride_d=}, {key.stride(1)=}"

    # Calculate the scale factor for attention
    sm_scale = 1.0 / (d_model ** 0.5)

    # Determine grid size
    def grid(META):
        return (
            triton.cdiv(n_ctx_q, META["BLOCK_M"])
            * triton.cdiv(n_ctx_k, META["BLOCK_N"]),
        )

    # Execute the kernel
    try:
        _score_kernel[grid](
            query,
            key,
            mask if mask is not None else torch.tensor([], device=device),
            out,
            n_ctx_q,
            n_ctx_k,
            d_model,
            query.stride(0),  # stride_ctx_q
            key.stride(0),  # stride_ctx_k
            stride_d,  # stride_d
            out.stride(0),  # stride_out_q
            out.stride(1),  # stride_out_k
            sm_scale,
            window_size=window_size,
        )
    except triton.CompilationError as e:
        print(f"Compilation error: {e}")
        print("Reducing block sizes and retrying...")
        # Reduce block sizes and retry
        _score_kernel.configs = [
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_D": 16},
                num_stages=5,
                num_warps=2,
            ),
        ]
        _score_kernel[grid](
            query,
            key,
            mask if mask is not None else torch.tensor([], device=device),
            out,
            n_ctx_q,
            n_ctx_k,
            d_model,
            query.stride(0),  # stride_ctx_q
            key.stride(0),  # stride_ctx_k
            stride_d,  # stride_d
            out.stride(0),  # stride_out_q
            out.stride(1),  # stride_out_k
            sm_scale,
            window_size=window_size,
        )

    return out
