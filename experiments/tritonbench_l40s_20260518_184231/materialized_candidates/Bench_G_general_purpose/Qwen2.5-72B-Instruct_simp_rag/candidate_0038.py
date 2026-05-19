import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_DMODEL": 32},
            num_stages=5,
            num_warps=2,
        ),
    ],
    key=["n_ctx_q", "n_ctx_k", "d_model"],
    prune_configs_by={
        "early_config_prune": early_config_prune,
        "perf_model": estimate_matmul_time,
        "top_k": 10,
    },
)
@triton.jit
def _score_kernel(
    q_ptr, k_ptr, m_ptr, out_ptr,
    n_ctx_q,
    n_ctx_k,
    d_model,
    stride_q_ctx, stride_k_ctx,
    stride_d,
    stride_out_ctx_q, stride_out_ctx_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    scale: tl.constexpr,
):
    # matrix multiplication
    pid = tl.program_id(0)

    # Determine the number of blocks in the grid
    grid_n = (n_ctx_k + BLOCK_N - 1) // BLOCK_N

    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    rm = tl.max_contiguous(tl.multiple_of(rm % n_ctx_q, BLOCK_M), BLOCK_M)
    rn = tl.max_contiguous(tl.multiple_of(rn % n_ctx_k, BLOCK_N), BLOCK_N)

    # Iterate through blocks of the d_model dimension and accumulate values into acc
    acc_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    rd = tl.arange(0, BLOCK_DMODEL)

    q_ptr_tile = q_ptr + (rm[:, None] * stride_q_ctx + rd[None, :] * stride_d)
    k_ptr_tile = k_ptr + (rd[:, None] * stride_d + rn[None, :] * stride_k_ctx)

    for d_max_offset in range(d_model, 0, -BLOCK_DMODEL):
        q_tile = tl.load(q_ptr_tile, mask=rd[None, :] < d_max_offset, other=0.0)
        k_tile = tl.load(k_ptr_tile, mask=rd[:, None] < d_max_offset, other=0.0)

        # In einsum notation, the following does: qd,dk->qk
        acc_tile += tl.dot(q_tile, k_tile)

        q_ptr_tile += BLOCK_DMODEL * stride_d
        k_ptr_tile += BLOCK_DMODEL * stride_d

    acc_tile *= scale

    if m_ptr is not None:
        m_tile = tl.load(m_ptr + (rm[:, None] * stride_out_ctx_q + rn[None, :] * stride_out_ctx_k), mask=(rm < n_ctx_q)[:, None] & (rn < n_ctx_k)[None, :], other=0.0)
        acc_tile += m_tile

    acc_tile = acc_tile.to(out_ptr.dtype.element_ty)

    # We rematerialize rm and rn here because it allows them to be deallocated above
    # instead of being kept in registers throughout the inner for-loop
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    out_offset_tile = rm[:, None] * stride_out_ctx_q + rn[None, :] * stride_out_ctx_k
    out_ptr_tile = out_ptr + out_offset_tile

    mask = (rm < n_ctx_q)[:, None] & (rn < n_ctx_k)[None, :]

    tl.store(out_ptr_tile, acc_tile, mask=mask)

def get_score(query, key, mask=None, scale=1.0):
    device = query.device

    # handle non-contiguous inputs if necessary
    if query.stride(0) > 1 and query.stride(1) > 1:
        query = query.contiguous()
    if key.stride(0) > 1 and key.stride(1) > 1:
        key = key.contiguous()
    if mask is not None and (mask.stride(0) > 1 or mask.stride(1) > 1):
        mask = mask.contiguous()

    # check constraints
    n_ctx_q, d_model = query.shape
    n_ctx_k, d_model_k = key.shape
    assert d_model == d_model_k, f"{query.shape=} {key.shape=}"

    # allocates output
    out = torch.empty((n_ctx_q, n_ctx_k), device=device, dtype=query.dtype)

    # Stride along the d_model dimension
    stride_d = query.stride(1)
    assert stride_d == key.stride(1), f"{stride_d=}, {key.stride(1)=}"

    # launch kernel
    def grid(META):
        return (
            triton.cdiv(n_ctx_q, META["BLOCK_M"])
            * triton.cdiv(n_ctx_k, META["BLOCK_N"]),
        )

    _score_kernel[grid](
        query,
        key,
        mask if mask is not None else torch.empty(0, device=device),
        out,
        n_ctx_q,
        n_ctx_k,
        d_model,
        query.stride(0),  # stride_q_ctx
        key.stride(0),  # stride_k_ctx
        stride_d,  # stride_d
        out.stride(0),  # stride_out_ctx_q
        out.stride(1),  # stride_out_ctx_k
        scale,
    )
    return out
